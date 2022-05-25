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



    def __init__(self, d_in , d_out, f_true, **kwargs):

        self.d_in = d_in

        self.d_out = d_out

        self.f_true = f_true

        self.x_min = util.dict_extract(kwargs, 'x_min', -1)
        if isinstance(self.x_min, int):
            self.x_min = [self.x_min for i in range(d_in)]

        self.x_max = util.dict_extract(kwargs, 'x_max', 1)
        if isinstance(self.x_max, int):
            self.x_max = [self.x_max for i in range(d_in)]

        self.n_samples = util.dict_extract(kwargs, 'n_samples', 64)

        self.noise_scale = util.dict_extract(kwargs, 'noise_scale', .5)

        self.n_val = util.dict_extract(kwargs, 'n_val', 256)

        self.resolution = util.dict_extract(kwargs, 'resolution', 540)


        # create training and valuation data
        self.y_train, self.x_train = self.create_training_data()
        self.y_val, self.x_val = self.create_valuation_data()

    

    def create_training_data(self):
        
        x_train = np.empty((self.n_samples, self.d_in))

        for i in range(self.d_in):
            if i == 0:
                x_train[:, i] = self._equi_data(self.n_samples, self.x_min[i], self.x_max[i])
            
            elif i == 1:
                x_train[:, i] = self._periodic_data(self.n_samples, self.x_min[i], self.x_max[i])
            
            else:
                x_train[:, i] = self._noise_data(self.n_samples, self.x_min[i], self.x_max[i])

        y_train = self.f_true(x_train) + np.random.normal(scale=1, size=(self.n_samples, self.d_out)) * self.noise_scale

        return util.to_tensor(y_train), util.to_tensor(x_train)



    def create_valuation_data(self):

        x_val = np.empty((self.n_samples, self.d_in))

        for i in range(self.d_in):
            # random evaluation points (possibly outside of [x_min, x_max])
            x_val[:, i] = self._noise_data(self.n_samples, self.x_min[i], self.x_max[i]) * np.random.normal(scale=1, size=self.n_samples)

        y_val = self.f_true(x_val) 

        return util.to_tensor(y_val), util.to_tensor(x_val)

    

    # def plot(self):