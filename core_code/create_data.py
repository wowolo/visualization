from inspect import signature
import numpy as np

import core_code.util as util
from core_code.util.default_config import init_config_data
from core_code.util.config_extractions import _f_true_fm



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


    def __init__(self, **config_data):

        self.config_data, self.all_losses = init_config_data(**config_data)
        self.all_losses = util.check_config(**config_data)



    def create_data(self, type):

        x = np.empty((0,self.config_data['d_in']))
        y = np.empty((0,self.config_data['d_out']))
        loss_activity = np.empty(0, dtype=int)

        for loss_num in self.all_losses:
            
            loss_config = util.extract_lossconfig(self.config_data, loss_num)
            loss_config = self.clean_1dbounds(loss_config)
            _x, _y, _loss_activity = self.loss_data_creator(type, loss_config, loss_num)
            x = np.concatenate([x, _x], axis=0)
            y = np.concatenate([y, _y], axis=0)
            loss_activity = np.concatenate([loss_activity, _loss_activity], axis=0, dtype=int)


        data_dict = {
            'x': util.to_tensor(x), 
            'y': util.to_tensor(y), 
            'loss_activity': loss_activity
        }
        
        return data_dict



    def loss_data_creator(self, type, loss_config, loss_num):

        if type == 'train':
            d_in = loss_config['d_in']
            n_samples = loss_config['n_train']
            x_min = loss_config['x_min_train']
            x_max = loss_config['x_max_train']
            data_generators = loss_config['data_generators_train']
            noise_scale = self.config_data['noise_scale']
        elif type == 'val':
            d_in = loss_config['d_in']
            n_samples = loss_config['n_val']
            x_min = loss_config['x_min_val']
            x_max = loss_config['x_max_val']
            data_generators = loss_config['data_generators_val']
            noise_scale = 0
        elif type == 'test':
            d_in = loss_config['d_in']
            n_samples = loss_config['n_test']
            x_min = loss_config['x_min_test']
            x_max = loss_config['x_max_test']
            data_generators = loss_config['data_generators_test']
            noise_scale = 0

            
        loss_x = np.empty((n_samples, d_in))

        if not(isinstance(data_generators, list)):
            data_generators = [data_generators]
        data_generators = data_generators[:d_in]

        if len(data_generators) < d_in: # default: fill missing data generators with last entry
            data_generators.append(data_generators[-1])
        

        temp_func_dict = {
            'equi': self._equi_data,
            'uniform': self._uniform_data,
            'periodic': self._periodic_data,
            'noise': self._noise_data
        }

        for d in range(d_in): # create data according to data generator in each dimension

            loss_x[:, d] = temp_func_dict[data_generators[d]](n_samples, x_min[d], x_max[d])
    
        # adjust function based on given focus_ind 
        if len(signature(_f_true_fm(self.config_data['f_true'])).parameters) == 1:
            f_true = lambda x: _f_true_fm(loss_config['f_true'])(x)
        else:
            f_true = lambda x: _f_true_fm(loss_config['f_true'])(x, loss_config['focus_ind'])

        loss_y = f_true(loss_x) + np.random.normal(scale=1, size=(n_samples, loss_config['d_out'])) * noise_scale

        _loss_activity = np.ones(loss_x.shape[0], dtype=int) * loss_num

        return loss_x, loss_y, _loss_activity
    

    # TODO make at initialization 
    @staticmethod
    def clean_1dbounds(config_data):
        for bound_key in ['x_min_train', 'x_max_train', 'x_min_val', 'x_max_val', 'x_min_test', 'x_max_test']:
            if isinstance(config_data[bound_key], int):
                config_data[bound_key] = [config_data[bound_key] for i in range(config_data['d_in'])]
        return config_data