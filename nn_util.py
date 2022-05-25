import torch
from torch.utils.data import TensorDataset, DataLoader

def report_hyperparam(key, hyperparam_dict):

    print('The following hyperparameters have been used for the construction of the {} \
        architecture:'.format(key))

    for paramkey in hyperparam_dict.keys():
        print('{}: {}'.format(paramkey, hyperparam_dict[paramkey]))
    


def DataGenerator(x_train, y_train, **kwargs):

    dataset = TensorDataset(x_train, y_train)

    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}

    data_generator =  DataLoader(dataset, **dataloader_dict)

    return data_generator