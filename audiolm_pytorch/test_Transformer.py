import os
# print(__name__)
# print(os.getcwd())
from audiolm_pytorch import Transformer
import torch


'''
self,
*,
dim,
depth,
heads,
dim_context = None,
cross_attend = False,
attn_dropout = 0.,
ff_dropout = 0.,
grad_shrink_alpha = 0.1,
cond_as_self_attn_prefix = False,
rel_pos_bias = True,
flash_attn = False,
**kwargs
'''

m = Transformer(dim=64, depth=4, heads=4)
'''
self,
x,
self_attn_mask = None,
context = None,
context_mask = None,
attn_bias = None,
return_kv_cache = False,
kv_cache = None
'''
input = torch.rand(8, 200, 64) # b, h, i, d
output = m(x=input)
print(output.shape)
