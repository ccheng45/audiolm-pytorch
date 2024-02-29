import torch
from hubert_kmeans import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer
from trainer import SemanticTransformerTrainer

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
dataset_folder= 'data/train-clean-100' 

wav2vec = HubertWithKmeans(
    checkpoint_path = hubert_ckpt,
    kmeans_path = hubert_quantizer
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()

trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 8,
    data_max_length = 320 * 32,
    num_train_steps = 140_000
)

trainer.train()


