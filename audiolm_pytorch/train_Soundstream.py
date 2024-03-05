
import torch
from soundstream import SoundStream
from trainer import SoundStreamTrainer

dataset_folder= 'data/' 

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

trainer = SoundStreamTrainer(
    soundstream,
    folder = dataset_folder,
    batch_size = 4,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length = 320 * 32,
    save_results_every = 100,
    save_model_every = 100,
    num_train_steps = 1000, 
    use_wandb_tracking=True
).cuda()
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason
with trainer.wandb_tracker(project = 'soundstream', run = 'baseline'):
    trainer.train()