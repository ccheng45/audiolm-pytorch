from collections import Counter
from typing_extensions import Annotated

from einops import rearrange, repeat, reduce

import torch
import torch.nn as nn

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from beartype.typing import Optional, Union, List

from data import SoundDataset, get_dataloader
from func_utils import * 

from hubert_kmeans import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer

folder= 'data/train-clean-100' 
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
batch_size = 4
drop_last = False
data_max_length =  320 * 32

def cycle(dl):
    while True:
        for data in dl:
            yield data

DATASET_FIELD_TYPE_CONFIG = dict(
    raw_wave = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    text = List[str],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

wav2vec = HubertWithKmeans(
    checkpoint_path = hubert_ckpt,
    kmeans_path = hubert_quantizer
)

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)


def has_duplicates(tup):
    counts = dict(Counter(tup))
    return any(filter(lambda count: count > 1, counts.values()))

ds = SoundDataset(
    folder,
    max_length = data_max_length,
    target_sample_hz = wav2vec.target_sample_hz,
    seq_len_multiple_of = wav2vec.seq_len_multiple_of
)
dl = get_dataloader(ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

def data_tuple_to_kwargs(data):
    ds_fields = None
    if not exists(ds_fields):
        ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
        assert not has_duplicates(ds_fields), 'dataset fields must not have duplicate field names'

    return dict(zip(ds_fields, data))
    
dl_iter = cycle(dl)
item=next(dl_iter) 
print(item)
out1 = data_tuple_to_kwargs(item)
print(out1['raw_wave'].shape)
raw_wave = out1['raw_wave']

### pass through hubert. 
semantic_token_ids = wav2vec(raw_wave, flatten = False)
print("wav2vec.codebook_size",wav2vec.codebook_size)
print("semantic_token_ids", semantic_token_ids.shape)
print("semantic_token_ids", semantic_token_ids)

semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')
print("after rearrange, semantic_token_ids", semantic_token_ids.shape)

transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
)

def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    eos_ids = torch.ones(1, device = device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1', b = b)
    ids = torch.cat((ids, eos_ids), dim = -1)
    return ids


semantic_token_ids = append_eos_id(semantic_token_ids, transformer.eos_id)
print("after append eos id, semantic_token_ids", semantic_token_ids.shape)
            
pad_id = -1

from torch.nn.utils.rnn import pad_sequence

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = pad_id)
print("after btach_unique_consecutive , semantic_token_ids", semantic_token_ids.shape)
 
input_ids = semantic_token_ids[:, :-1]
print("input_ids", input_ids.shape)

mask_prob =0.15      
self_attn_mask = generate_mask_with_prob(input_ids.shape, mask_prob, input_ids.device)
print("self_attn_mask.shape", self_attn_mask.shape)
print("self_attn_mask", self_attn_mask)

logits = transformer(
        ids = input_ids,
        text = None,
        text_embeds = None,
        self_attn_mask = self_attn_mask,    
    )        
print("logits shape:", logits.shape)
logits = rearrange(logits, 'b n c -> b c n')
print("logits after rearrange", logits.shape)
print("semantic_token_ids", semantic_token_ids.shape)
loss = F.cross_entropy(
    logits,
    semantic_token_ids,
    ignore_index = -1
)
print(loss)
