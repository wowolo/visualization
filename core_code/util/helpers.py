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
    return torch.Tensor(x)



def make_jsonable(x):
    try:
        json.dumps(x)
        return x
    except:
        return str(x)



def create_config(kwargs, default_extraction_strings):

    config = {string: None for string in default_extraction_strings}

    for string in default_extraction_strings:
        
        if string in kwargs.keys():
            item = kwargs[string]
        else:
            item = default_extraction_strings[string]
        
        config[string] = item
    
    return config



def check_config(**kwargs):
    all_losses = set()
    # check values either non dict or dict with loss_#num format
    # + check that values
    for val in kwargs.values():
        temp_set = set()
        if isinstance(val, dict):
            for key in val.keys():
                sep_key = key.split('_')
                if (len(sep_key) != 2) or (sep_key[0] != 'loss') or (sep_key[1] != str(int(sep_key[1]))):
                    raise ValueError('The format in the configuration is invalid.')
                num_loss = int(sep_key[1])
                temp_set.update({num_loss})
        if len(all_losses) == 0:
            all_losses = temp_set
        elif len(temp_set) > 0:
            assert all_losses == temp_set

    if len(all_losses) == 0:
        all_losses = {0}

    return all_losses



def extract_lossconfig(config, loss):
        loss_config = dict.fromkeys(config.keys())
        loss_key = 'loss_' + str(loss)

        for key in loss_config.keys():
            value = config[key]
            if isinstance(value, dict):
                value = value[loss_key]
            loss_config[key] = value

        return loss_config



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




class dimred_MSELoss(torch.nn.Module):

    def __init__(self, dimension_activity):

        super(dimred_MSELoss,self).__init__()
        self.dimension_activity = dimension_activity
    

    
    def forward(self, output, target):

        if output.shape[0] == 0: return 0

        dimred_output = output[:, self.dimension_activity]
        dimred_target = target[:, self.dimension_activity]

        return torch.sum((dimred_output - dimred_target)**2)