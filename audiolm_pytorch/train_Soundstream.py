
import torch
from soundstream import SoundStream
from trainer import SoundStreamTrainer

dataset_folder= 'data/' 

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
    rq_groups = 2,                       # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
    attn_window_size = 128,              # local attention receptive field at bottleneck
    attn_depth = 2  
)

trainer = SoundStreamTrainer(
    soundstream,
    folder = dataset_folder,
    batch_size = 8,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length = 320 * 32,
    save_results_every = 2000,
    save_model_every = 2000,
    num_train_steps = 10001, 
    use_wandb_tracking=True
).cuda()

# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason
with trainer.wandb_tracker(project = 'soundstream', run = 'a100-2'):
    trainer.train()