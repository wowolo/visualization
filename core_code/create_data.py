from inspect import signature
from functools import wraps
import numpy as np
from torch.utils.data import Dataset, DataLoader

import core_code.util as util

###############################################################################
############# create data
###############################################################################





class CreateData():


    @staticmethod
    def _equi_data(n_samples, x_min_i, x_max_i):
        return np.linspace(x_min_i, x_max_i , n_samples)



    @staticmethod
    def _uniform_data(n_samples, x_min_i, x_max_i):
        return np.random.rand(n_samples) * (x_max_i - x_min_i) + x_min_i



    @staticmethod
    def _noise_data(n_samples, x_min_i, x_max_i):
        return np.random.normal(size=n_samples) % (x_max_i - x_min_i) + (x_max_i + x_min_i) / 2



    @staticmethod
    def _periodic_data(n_samples, x_min_i, x_max_i):
        samples_1 = int(n_samples/2)
        per_1 = np.sin(1.5 * np.pi * np.linspace(-1, 1, int(n_samples/2))) * (x_max_i - x_min_i) + x_min_i
        rest_samples = n_samples - samples_1
        per_2 = - np.sin(1.5 * np.pi * np.linspace(-1, 1, rest_samples)) * (x_max_i - x_min_i) + x_min_i
        data = np.concatenate((per_1, per_2))
        return data

    

    @staticmethod
    def _f_true_fm(value):
        f_true = util.function_library(value)
        return f_true




    def clean_1dbounds(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for bound_key in ['x_min', 'x_max']:
                if isinstance(self.config[bound_key], int):
                    self.config[bound_key] = [self.config[bound_key] for i in range(self.config['d_in'])]
            return func(self, *args, **kwargs)
        return wrapper



    def __init__(self, **kwargs):

        self.config = self.init_config(**kwargs)
        
        # # create training and valuation data
        # self.y_train, self.x_train = self.create_training_data()
        # self.y_val, self.x_val = self.create_valuation_data()


    
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
            'n_val': 128,
            'data_generators': 'equi',
        }

        config = {string: None for string in default_extraction_strings}

        for string in default_extraction_strings:
            
            if string in kwargs.keys():
                item = kwargs[string]
            else:
                item = default_extraction_strings[string]
            
            config[string] = item
        
        return config

    

    @clean_1dbounds
    def create_training_data(self):
        
        d_in = self.config['d_in']
        n_samples = self.config['n_samples']
        x_min = self.config['x_min']
        x_max = self.config['x_max']

        x_train = np.empty((n_samples, self.config['d_in']))

        data_generators = self.config['data_generators']
        if not(isinstance(data_generators, list)):
            data_generators = [data_generators]
        data_generators = data_generators[:d_in]

        if len(data_generators) < d_in: # default: fill missing data generators with last entry
            data_generators.append(data_generators[-1])
        

        for d in range(d_in): # create data according to data generator in each dimension

            if data_generators[d] == 'equi':
                x_train[:, d] = self._equi_data(n_samples, x_min[d], x_max[d])
            
            elif data_generators[d] == 'uniform':
                x_train[:, d] = self._uniform_data(n_samples, x_min[d], x_max[d])

            elif data_generators[d] == 'periodic':
                x_train[:, d] = self._periodic_data(n_samples, x_min[d], x_max[d])
            
            elif data_generators[d] == 'noise':
                x_train[:, d] = self._noise_data(n_samples, x_min[d], x_max[d])
    
        # adjust function based on given focus_ind 
        if len(signature(self._f_true_fm(self.config['f_true'])).parameters) == 1:
            f_true = lambda x: self._f_true_fm(self.config['f_true'])(x)
        else:
            f_true = lambda x: self._f_true_fm(self.config['f_true'])(x, self.config['focus_ind'])

        y_train = f_true(x_train) + np.random.normal(scale=1, size=(n_samples, self.config['d_out'])) * self.config['noise_scale']

        return util.to_tensor(x_train), util.to_tensor(y_train)



    @clean_1dbounds
    def create_valuation_data(self):

        d_in = self.config['d_in']
        n_val = self.config['n_val']
        x_min = self.config['x_min']
        x_max = self.config['x_max']

        x_val = np.empty((n_val, d_in))

        for i in range(d_in):
            # random validation points (possibly outside of [x_min, x_max]) - dependent on stretch
            stretch = 1
            center = (x_max[i] + x_min[i]) * 0.5
            temp_x_min = center + stretch * (x_min[i] - center)
            temp_x_max = center + stretch * (x_max[i] - center)
            x_val[:, i] = self._equi_data(n_val, temp_x_min, temp_x_max)

        # adjust function based on given focus_ind 
        if len(signature(self._f_true_fm(self.config['f_true'])).parameters) == 1:
            f_true = lambda x: self._f_true_fm(self.config['f_true'])(x)
        else:
            f_true = lambda x: self._f_true_fm(self.config['f_true'])(x, self.config['focus_ind'])
            
        y_val = f_true(x_val) 

        return util.to_tensor(x_val), util.to_tensor(y_val)

    

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