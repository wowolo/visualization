import numpy as np

def dict_extract(kwargs, key, default=None):
    
    if key in kwargs.keys():
        key_val = kwargs[key]
    else:
        key_val = default

    return key_val