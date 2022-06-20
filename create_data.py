import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import util

###############################################################################
############# create data
###############################################################################





class CreateData():



    @staticmethod
    def _equi_data(n_samples, x_min_i, x_max_i):
        return np.linspace(x_min_i, x_max_i , n_samples)



    @staticmethod
    def _noise_data(n_samples, x_min_i, x_max_i):
        return np.random.normal(size=n_samples) % (x_max_i - x_min_i) + (x_max_i + x_min_i) / 2



    @staticmethod
    def _uniform_data(n_samples, x_min_i, x_max_i):
        return np.random.rand(n_samples) * (x_max_i - x_min_i) + x_min_i



    @staticmethod
    def _periodic_data(n_samples, x_min_i, x_max_i):
        samples_1 = int(n_samples/2)
        per_1 = np.sin(1.5 * np.pi * np.linspace(-1, 1, int(n_samples/2))) * (x_max_i - x_min_i) + x_min_i
        rest_samples = n_samples - samples_1
        per_2 = - np.sin(1.5 * np.pi * np.linspace(-1, 1, rest_samples)) * (x_max_i - x_min_i) + x_min_i
        data = np.concatenate((per_1, per_2))
        return data



    def __init__(self, **kwargs):

        self.config = self.init_config(**kwargs)

        if isinstance(self.config['x_min'], int):
            self.config['x_min'] = [self.config['x_min'] for i in range(self.config['d_in'])]
        if isinstance(self.config['x_max'], int):
            self.config['x_max'] = [self.config['x_max'] for i in range(self.config['d_in'])]

        # create training and valuation data
        self.y_train, self.x_train = self.create_training_data()
        self.y_val, self.x_val = self.create_valuation_data()


    
    def init_config(self, **kwargs):

        default_extraction_strings = {
            'd_in': None, 
            'd_out': None, 
            'f_true': None,
            'focus_ind': 0, 
            'x_min': -1, 
            'x_max': 1, 
            'n_samples': 256, 
            'noise_scale': .1,
            'n_val': 128
        }

        config = {string: None for string in default_extraction_strings}

        for string in default_extraction_strings:
            
            if string in kwargs.keys():
                item = kwargs[string]
            else:
                item = default_extraction_strings[string]
            
            config[string] = item
        
        return config

    

    def create_training_data(self):
        
        n_samples = self.config['n_samples']
        x_min = self.config['x_min']
        x_max = self.config['x_max']

        x_train = np.empty((n_samples, self.config['d_in']))

        for i in range(self.config['d_in']):
            if i == 0:
                x_train[:, i] = self._equi_data(n_samples, x_min[i], x_max[i])
            
            elif i == 1:
                x_train[:, i] = self._periodic_data(n_samples, x_min[i], x_max[i])
            
            else:
                x_train[:, i] = self._noise_data(n_samples, x_min[i], x_max[i])
        
        # adjust function based on given focus_ind 
        f_true = lambda x: self.config['f_true'](x, self.config['focus_ind'])
        y_train = f_true(x_train) + np.random.normal(scale=1, size=(n_samples, self.config['d_out'])) * self.config['noise_scale']

        return util.to_tensor(y_train), util.to_tensor(x_train)



    def create_valuation_data(self):
        n_val = self.config['n_val']
        x_min = self.config['x_min']
        x_max = self.config['x_max']
        d_in = self.config['d_in']

        x_val = np.empty((n_val, d_in))

        for i in range(d_in):
            # random evaluation points (possibly outside of [x_min, x_max])
            stretch = 1
            center = (x_max[i] + x_min[i]) * 0.5
            temp_x_min = center + stretch * (x_min[i] - center)
            temp_x_max = center + stretch * (x_max[i] - center)
            x_val[:, i] = self._equi_data(n_val, temp_x_min, temp_x_max)

        # adjust function based on given focus_ind 
        f_true = lambda x: self.config['f_true'](x, self.config['focus_ind'])
        y_val = f_true(x_val) 

        return util.to_tensor(y_val), util.to_tensor(x_val)

    

class CustomDataset(Dataset):

    def __init__(self, x_train, y_train, loss_activity):
        
        self.x_train = x_train
        self.y_train = y_train
        self.loss_activity = loss_activity



    def __len__(self):
        return self.x_train.shape[0]



    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.loss_activity[idx]



def DataGenerators(x_train, y_train, loss_activity, **kwargs):

    # check for tuple input -> loss activity

    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}
    dataloader_dict['batch_size']  = util.dict_extract(dataloader_dict, 'batch_size', 64)
    dataloader_dict['shuffle']  = util.dict_extract(dataloader_dict, 'shuffle', True)

    bool_separate_loss_batching = util.dict_extract(kwargs, 'separate_loss_batching', True) # default value
    bool_print_datagen_config = util.dict_extract(kwargs, 'print_datagen_config', True) # default value

    # structure the generator by shuffle and separate_loss_batching
    if bool_separate_loss_batching:
        # determine the ratios based on given loss_activity and “total” batch size
        _total_activities = [(loss_activity == i).sum() for i in range(1, int(loss_activity.max()) + 1)]
        _ratios = [float(_total_activities[i] / sum(_total_activities)) for i in range(len(_total_activities))]
        _max_ratio = np.argmax(_ratios)

        # guarantee that batch size is sufficiently large to sample according to non-zero ratios
        _min_batch_size = sum([ratio > 0 for ratio in _ratios])
        if dataloader_dict['batch_size'] < _min_batch_size:
            raise ValueError("Since 'separate_loss_batching' is True and the loss_activity indicates that {} losses are used we need a total 'batch_size' of at least {}.".format(_min_batch_size, _min_batch_size))
        
        _batch_sizes = [max(1, int(ratio * dataloader_dict['batch_size'])) for ratio in _ratios]
        _batch_sizes[_max_ratio] = dataloader_dict['batch_size'] - sum(_batch_sizes[:_max_ratio] + _batch_sizes[_max_ratio+1:])
        _ind_lossdatas = [(loss_activity == i) for i in range(1, int(loss_activity.max()) + 1)]
        _dataset_partitions = [CustomDataset(x_train[_ind_lossdatas[i]], y_train[_ind_lossdatas[i]], loss_activity[_ind_lossdatas[i]]) for i in range(len(_ind_lossdatas))]
        data_generators = [DataLoader(_dataset_partitions[i], batch_size=_batch_sizes[i], shuffle=dataloader_dict['shuffle']) for i in range(len(_ind_lossdatas))]
        
        if bool_print_datagen_config:
            # print the configuration with ratios
            min_iter =  min([data_generators[i].__len__() for i in range(len(data_generators))])
            print('The following configuration has been used for the construction of the training loops:')
            print('Data partition based on losses: {}'.format(bool_separate_loss_batching))
            print('Number of losses considered: {}'.format(int(loss_activity.max())))
            print('Data ratio for each partition: {}'.format(_ratios))
            print('Total batch size: {}'.format(dataloader_dict['batch_size']))
            print('Derived sub-batch sizes for each partition: {}'.format(_batch_sizes))
            print('Number of iterations in one epoch: {}'.format(min_iter))
            print('Shuffle: {}'.format(dataloader_dict['shuffle']))

    
    else:
        dataset = CustomDataset(x_train, y_train, loss_activity)
        data_generators =  [DataLoader(dataset, **dataloader_dict)]
        if bool_print_datagen_config:
            # print the configuration with ratios
            min_iter =  min([data_generators[i].__len__() for i in range(len(data_generators))])
            print('The following configuration has been used for the construction of the training loops:')
            print('Data partition based on losses: {}'.format(bool_separate_loss_batching))
            print('Number of losses considered: {}'.format(int(loss_activity.max())))
            print('Total batch sizes: {}'.format(dataloader_dict['batch_size']))
            print('Number of iterations in one epoch: {}'.format(min_iter))
            print('Shuffle: {}'.format(dataloader_dict['shuffle']))

    return data_generators