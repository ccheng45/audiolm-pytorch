from collections import Counter
from typing_extensions import Annotated


import torch
import torch.nn as nn

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from beartype.typing import Optional, Union, List

from data import SoundDataset, get_dataloader
from func_utils import * 

from hubert_kmeans import HubertWithKmeans

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



### pass through hubert. 

