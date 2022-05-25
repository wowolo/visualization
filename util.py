import numpy as np
import torch

def dict_extract(kwargs, key, default=None):
    
    if key in kwargs.keys():
        key_val = kwargs[key]
    else:
        key_val = default

    return key_val

def to_tensor(x):
    return torch.Tensor(x).double()