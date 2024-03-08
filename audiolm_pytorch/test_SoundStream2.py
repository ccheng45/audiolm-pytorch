from soundstream import SoundStream
import torchaudio
import torch
from einops import rearrange, repeat, reduce

path = 'results/soundstream.20000.pt' 
soundstream = SoundStream.init_and_load_from(path).to("cuda")

audio = torch.randn(1, 512 * 320).to('cuda')

codes = soundstream.tokenize(audio)

# you can now train anything with the codebook ids
recon_audio_from_codes = soundstream.decode_from_codebook_indices(codes)

# sanity check

print(
    torch.allclose(recon_audio_from_codes, soundstream(audio, return_recons_only=True))
)
