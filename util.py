import json
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


def make_jsonable(x):
    try:
        json.dumps(x)
        return x
    except:
        return str(x)


def dict_to_file(dict, file_path, format='v'):
    # format: 'v' or 'h'
    with open(file_path, 'w') as file:
        if format == 'v':
            for key, val in dict.items():
                file.write('{}: {}\n'.format(key, val))
        else:
            json_dict = {key: make_jsonable(dict[key]) for key in dict.keys()}
            file.write(json.dumps(json_dict))


def dictvals_to_list(dict):
    for key, val in dict.items():
        if not(isinstance(val, list)):
            dict[key] = [val] 

    return dict


# class create_loss_ratio
# Bernoulli 0/1 - whether dimensional loss is active or inactive
# input d_out, Bernoulli p (+ draw according to number of training steps - locally or globally?)