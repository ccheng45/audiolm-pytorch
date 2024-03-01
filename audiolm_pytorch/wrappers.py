import torch
import torch.nn as nn
from torch import einsum, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from einops import rearrange, repeat, reduce

from beartype.typing import Optional, Union, List
from beartype import beartype

from audiolm_pytorch import SemanticTransformer, CoarseTransformer
from soundstream import SoundStream
from encodec_wrapper import EncodecWrapper
from vq_wav2vec import FairseqVQWav2Vec
from hubert_kmeans import HubertWithKmeans
from utils import AudioConditionerBase
from func_utils import * 


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    eos_ids = torch.ones(1, device = device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1', b = b)
    ids = torch.cat((ids, eos_ids), dim = -1)
    return ids

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

class SemanticTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: SemanticTransformer,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        pad_id = -1,
        unique_consecutive = True,
        mask_prob = 0.15
    ):
        super().__init__()
        print("SemanticTransformerWrapper init")
        self.wav2vec = wav2vec
        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        assert not exists(self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id
        self.eos_id = transformer.eos_id
        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_text(self, text):
        return self.transformer.embed_text(text, output_device = self.device)

    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        max_length,
        text: Optional[List[str]] = None,
        text_embeds = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_ids = None,
        batch_size = 1,
        cond_scale = 3,
        filter_thres = 0.9,
        temperature = 1.,
        use_kv_cache = True,
        include_eos_in_output = True,  # if doing hierarchical sampling, eos must be kept for an easy time
        **kwargs
    ):
        device = self.device

        # derive wav2vec ids from the input wave

        if exists(prime_wave):
            assert not exists(prime_ids)
            assert exists(self.wav2vec)
            ids = self.wav2vec(
                prime_wave,
                flatten = False,
                input_sample_hz = prime_wave_input_sample_hz
            )
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

        if self.unique_consecutive:
            ids = batch_unique_consecutive(ids, pad_value = self.pad_id)

        # derive joint audio-text embeddings if needed

        if exists(self.audio_conditioner) and exists(prime_wave):
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = prime_wave, namespace = 'semantic')

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # start length and get running id output

        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()

        last_logit_indices = (ids != self.pad_id).sum(dim = -1).long()

        # kv cache

        kv_cache = None
        logits = None

        # sample from transformer

        for ind in tqdm(range(start_length, max_length), desc = 'generating semantic'):

            new_logits, new_kv_cache = self.transformer.forward_with_cond_scale(
                ids = sample_semantic_ids,
                text_embeds = text_embeds,
                cond_scale = cond_scale,
                kv_cache = kv_cache,
                return_kv_cache = True,
                **kwargs
            )

            if use_kv_cache:
                kv_cache = new_kv_cache
                logits = safe_cat(logits, new_logits, dim = -2)
            else:
                logits = new_logits

            last_logit_indices_expanded = repeat(last_logit_indices, 'b -> b 1 c', b = batch, c = logits.shape[-1])
            last_logits = logits.gather(1, last_logit_indices_expanded)

            last_logits = rearrange(last_logits, 'b 1 c -> b c')

            filtered_logits = top_k(last_logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            sample_semantic_ids = torch.cat((sample_semantic_ids, sampled), dim = -1)

            if all_rows_have_eos_id(sample_semantic_ids, self.eos_id):
                break

            last_logit_indices += 1

        sample_semantic_ids = mask_out_after_eos_id(sample_semantic_ids, self.eos_id, keep_eos = False)

        return sample_semantic_ids

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        text = None,
        text_embeds = None,
        return_loss = False,
        **kwargs
    ):
        print("-------------------------forward!!!!!!")
        print("raw_wave", raw_wave.shape)
        assert exists(raw_wave), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'semantic')

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(semantic_token_ids, self.transformer.eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = self.pad_id)

        input_ids = semantic_token_ids
        if return_loss:
            input_ids = semantic_token_ids[:, :-1]

        self_attn_mask = None
        if self.mask_prob > 0. and self.training:
            self_attn_mask = generate_mask_with_prob(input_ids.shape, self.mask_prob, input_ids.device)
        '''
        self,
        *,
        ids = None,
        return_loss = False,
        text: Optional[List[str]] = None,
        text_embeds = None,
        self_attn_mask = None,
        cond_drop_prob = None,
        unique_consecutive = None,
        kv_cache = None,
        return_kv_cache = False
        '''
        logits = self.transformer(
            ids = input_ids,
            text = text,
            text_embeds = text_embeds,
            self_attn_mask = self_attn_mask,
            **kwargs
        )        
        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            semantic_token_ids,
            ignore_index = self.pad_id
        )
        return loss



class CoarseTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: CoarseTransformer,
        codec: Optional[Union[SoundStream, EncodecWrapper]]  = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        pad_id = -1,
        unique_consecutive = True,
        semantic_cross_entropy_loss_weight = 1.,
        mask_prob = 0.15
    ):
        super().__init__()
        self.codec = codec
        self.wav2vec = wav2vec

        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.semantic_cross_entropy_loss_weight = semantic_cross_entropy_loss_weight

        self.num_coarse_quantizers = transformer.num_coarse_quantizers * codec.rq_groups
        self.semantic_eos_id = transformer.semantic_eos_id
        self.coarse_eos_id = transformer.coarse_eos_id

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        semantic_token_ids,
        prime_wave: Optional[Tensor] = None,
        prime_wave_input_sample_hz = None,
        prime_coarse_token_ids: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        max_time_steps = 512,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
        use_kv_cache = True,
        **kwargs
    ):
        batch, device = semantic_token_ids.shape[0], self.device

        semantic_token_ids = semantic_token_ids.to(device)

        # initialize coarse token ids
        # if a prime audio wave was supplied, then start off with appropriate acoustic tokens

        assert not (exists(prime_wave) and exists(prime_coarse_token_ids)), 'you can either pass in the prime as a raw wave (codec required) or as preprocessed acoustic token ids'

        if exists(prime_coarse_token_ids):
            coarse_token_ids = prime_coarse_token_ids
        elif exists(prime_wave):
            assert exists(self.codec)
            with torch.inference_mode():
                self.codec.eval()

                _, indices, _ = self.codec(
                    prime_wave,
                    return_encoded = True,
                    input_sample_hz = prime_wave_input_sample_hz
                )

                coarse_token_ids = indices[..., :self.num_coarse_quantizers]
                coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        else:
            coarse_token_ids = torch.empty((batch, 0), device = device, dtype = torch.long)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value=self.pad_id)

        # initialize

        init_coarse_time_step = 0
        sampled_coarse_token_ids = coarse_token_ids.clone()

        # kv cache

        kv_cache = None
        embed_cache = None

        for time_step in tqdm(range(init_coarse_time_step, max_time_steps), desc = 'generating coarse'):
            for ind in range(self.num_coarse_quantizers):
                just_finished_quantizer_step = (ind == 0 and time_step > 0)

                (_, coarse_logits), (next_kv_cache, next_embed_cache) = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = sampled_coarse_token_ids,
                    semantic_token_ids = semantic_token_ids,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_kv_cache = True,
                    kv_cache = kv_cache,
                    embed_cache = embed_cache,
                    return_only_coarse_logits = True,
                    **kwargs
                )

                if use_kv_cache:
                    kv_cache = next_kv_cache
                    embed_cache = next_embed_cache

                last_coarse_logits = coarse_logits[:, -1]

                if not just_finished_quantizer_step:
                    last_coarse_logits[:, -1] = float('-inf') # prevent from eos in the middle of a time step

                filtered_logits = top_k(last_coarse_logits, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_coarse_token_ids = torch.cat((sampled_coarse_token_ids, sampled), dim = -1)

        sampled_coarse_token_ids = mask_out_after_eos_id(sampled_coarse_token_ids, self.coarse_eos_id, keep_eos = False)
        sampled_coarse_token_ids = rearrange(sampled_coarse_token_ids, 'b (n q) -> b n q', q = self.num_coarse_quantizers)

        if not reconstruct_wave:
            return sampled_coarse_token_ids

        assert exists(self.codec)

        coarse_tokens_are_variable_lengthed = (sampled_coarse_token_ids == -1).any()

        if not coarse_tokens_are_variable_lengthed:
            wav = self.codec.decode_from_codebook_indices(sampled_coarse_token_ids)
            return rearrange(wav, 'b 1 n -> b n')

        # handle variable lengthed coarse tokens

        wavs = []
        for coarse_sample in sampled_coarse_token_ids:
            has_padding = reduce(coarse_sample == -1, 'n q -> n', 'any')
            coarse_sample_without_padding = coarse_sample[~has_padding]

            if has_padding.all():
                wavs.append(None)
                continue

            coarse_sample_without_padding = rearrange(coarse_sample_without_padding, '... -> 1 ...')

            wav = self.codec.decode_from_codebook_indices(coarse_sample_without_padding)
            wav = rearrange(wav, '1 1 n -> n')

            wavs.append(wav)

        return wavs

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        raw_wave_for_codec = None,
        text = None,
        text_embeds = None,
        coarse_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        raw_wave_for_codec = default(raw_wave_for_codec, raw_wave)
        assert exists(raw_wave_for_codec) or exists(coarse_token_ids), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        assert not all(map(exists, (raw_wave, raw_wave_for_codec, semantic_token_ids, coarse_token_ids)))

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'coarse') # technically audio embeds, but shared text-audio joint embedding space for mulan

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

        if not exists(coarse_token_ids):
            assert exists(self.codec), 'Codec must be provided if given raw wave for training'

            with torch.inference_mode():
                self.codec.eval()
                _, indices, _ = self.codec(raw_wave_for_codec, return_encoded = True)

                batch, num_timesteps = raw_wave_for_codec.shape
                num_frames = int(num_timesteps / self.codec.seq_len_multiple_of)

                assert indices.shape[0] == batch and indices.shape[1] == num_frames, \
                    f'Expected indices to have shape (batch, num_frames, num_coarse_quantizers + num_fine_quantizers), but got {indices.shape}'

                coarse_token_ids = indices[..., :self.num_coarse_quantizers]

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')
        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(semantic_token_ids, self.transformer.semantic_eos_id)
            coarse_token_ids = append_eos_id(coarse_token_ids, self.transformer.coarse_eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = self.pad_id)

        if return_loss:
            semantic_labels, coarse_labels = semantic_token_ids, coarse_token_ids.clone()
            coarse_token_ids = coarse_token_ids[:, :-1]

        # self attention mask would omit any padding and eos tokens in the semantic prime

        self_attn_mask = (semantic_token_ids != self.pad_id) & (semantic_token_ids != self.semantic_eos_id)
        semantic_token_ids = semantic_token_ids.masked_fill(~self_attn_mask, 0)

        coarse_token_len = coarse_token_ids.shape[-1]
        self_attn_mask = F.pad(self_attn_mask, (1, coarse_token_len + 1), value = True) # attend to semantic bos and all coarse tokens

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(self_attn_mask.shape, self.mask_prob, device = self_attn_mask.device)

        semantic_logits, coarse_logits = self.transformer(
            semantic_token_ids = semantic_token_ids,
            coarse_token_ids = coarse_token_ids,
            self_attn_mask = self_attn_mask,
            text = text,
            text_embeds = text_embeds,
            **kwargs
        )

        # whether to early return the logits

        if not return_loss:
            return semantic_logits, coarse_logits

        coarse_logits, semantic_logits = map(lambda t: maybe(rearrange)(t, 'b n c -> b c n'), (coarse_logits, semantic_logits))

        if self.unique_consecutive:
            num_coarse_logits, _num_semantic_logits = coarse_labels.numel(), (semantic_labels != self.pad_id).sum()
        else:
            num_coarse_logits, _num_semantic_logits = coarse_logits.shape[-1], semantic_logits.shape[-1]

        semantic_loss = 0.
        num_semantic_logits = 0

        if self.semantic_cross_entropy_loss_weight > 0 and exists(semantic_logits):
            num_semantic_logits = _num_semantic_logits

            semantic_loss = F.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index = self.pad_id
            )

        coarse_loss = F.cross_entropy(
            coarse_logits,
            coarse_labels,
            ignore_index = self.pad_id
        )

        return (
            semantic_loss * num_semantic_logits * self.semantic_cross_entropy_loss_weight +
            coarse_loss * num_coarse_logits
        ) / (num_semantic_logits + num_coarse_logits)
