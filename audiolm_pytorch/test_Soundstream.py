
from soundstream import SoundStream
import torchaudio
import torch
from einops import rearrange, repeat, reduce
# soundstream = SoundStream(
#     codebook_size = 1024,
#     rq_num_quantizers = 8,
# )

path = 'results/soundstream.444.pt' 
model = SoundStream.init_and_load_from(path).to("cuda")

x, sr = torchaudio.load('input.wav')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000

x = x.to("cuda")
print("x", x.shape)
with torch.no_grad():
    # y = model.encode(x)
    y = model.tokenize(x)
    print("y shape", y.shape)
    
    z = model.decode_from_codebook_indices(y)
    print("z", z.shape)
    # y = y[:, :, :4]  # if you want to reduce code size.
    # z = model.decode(y)
    
    z = rearrange(z, '1 1 n -> 1 n')
    print("z_flat", z.shape)
    z = z.to("cpu")
    torchaudio.save('output.wav',z, sr)