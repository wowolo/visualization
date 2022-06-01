import torch
from torch.utils.data import TensorDataset, DataLoader



def report_config(config):

    print('The following configuration has been used for the construction of the {} \
        architecture:'.format(config['architecture_key']))

    for paramkey in config.keys():
        print('{}: {}'.format(paramkey, config[paramkey]))
    


def DataGenerator(x_train, y_train, **kwargs):

    dataset = TensorDataset(x_train, y_train)

    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}

    data_generator =  DataLoader(dataset, **dataloader_dict)

    return data_generator


def create_config(kwargs, default_extraction_strings):

    config_architecture = {string: None for string in default_extraction_strings}

    for string in default_extraction_strings:
        
        if string in kwargs.keys():
            item = kwargs[string]
        else:
            item = default_extraction_strings[string]
        
        config_architecture[string] = item
    
    return config_architecture