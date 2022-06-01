import numpy as np

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

        y_train = self.config['f_true'](x_train) + np.random.normal(scale=1, size=(n_samples, self.config['d_out'])) * self.config['noise_scale']

        return util.to_tensor(y_train), util.to_tensor(x_train)



    def create_valuation_data(self):
        n_samples = self.config['n_samples']
        x_min = self.config['x_min']
        x_max = self.config['x_max']
        d_in = self.config['d_in']

        x_val = np.empty((n_samples, d_in))

        for i in range(d_in):
            # random evaluation points (possibly outside of [x_min, x_max])
            x_val[:, i] = self._noise_data(n_samples, x_min[i], x_max[i]) * np.random.normal(scale=1, size=n_samples)

        y_val = self.config['f_true'](x_val) 

        return util.to_tensor(y_val), util.to_tensor(x_val)

    

    # def plot(self):