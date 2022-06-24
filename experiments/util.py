import os
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import ParameterGrid



class BasicManager():
    
    @staticmethod
    def make_jsonable(x):

        try:
            json.dumps(x)
            return x
        except:
            return str(x)



    @staticmethod
    def dictvals_to_list(dict):

        for key, val in dict.items():
            if not(isinstance(val, list)):
                dict[key] = [val] 

        return dict


    
    @staticmethod
    def set_randomness(torch_seed, numpy_seed):
        import torch
        torch.manual_seed(torch_seed)
        import numpy as np
        np.random.seed(numpy_seed)



    @staticmethod
    def default_experimentbatch_name(experimentbatch_name):

        if isinstance(experimentbatch_name, type(None)):
            timestamp = datetime.now().strftime('%H-%M_%d.%m.%Y')
            experimentbatch_name = 'experiments_{}'.format(timestamp)
        
        return experimentbatch_name

    
    
    def default_experimentbatch_dir(self, root, experimentbatch_name):

        root = Path(root)
        experimentbatch_name = self.default_experimentbatch_name(experimentbatch_name)
        experimentbatch_path = root / experimentbatch_name

        i = 0
        while True:
            try:
                os.mkdir(experimentbatch_path)
                break
            except FileExistsError:
                experimentbatch_path = root / (experimentbatch_name + '_{}'.format(i))
                i += 1
                continue

        return experimentbatch_path

    

    def dict_to_file(self, dict, file_path, format='v'):
        # format: 'v' or 'h'
        with open(file_path, 'w') as file:
            if format == 'v':
                for key, val in dict.items():
                    file.write('{}: {}\n'.format(key, val))
            else:
                json_dict = {key: self.make_jsonable(dict[key]) for key in dict.keys()}
                file.write(json.dumps(json_dict))

    

    def list_to_file(self, list_, file_path, format='v'):

        with open(file_path, 'w') as file:
            file.write(json.dumps(list_))


    
    @staticmethod
    def save_network_weights(experiment_path, nn_model):

        nn_model.save(experiment_path / 'nn_model_weights.pt')



    @staticmethod
    def create_storage_dir(storage_path, file_path_parent):

        if isinstance(storage_path, type(None)):
            storage_path = file_path_parent / 'storage'
        else:
            storage_path = Path(storage_path)

        try:
            os.mkdir(storage_path)
        except FileExistsError:
            pass

        return storage_path



    @staticmethod
    def create_experiment_dir(experimentbatch_path, i, print_only=False):

        exp_ind = 'experiment_{}'.format(i)
        experiment_path = experimentbatch_path / exp_ind

        if not(print_only):
            os.mkdir(experiment_path)

        return experiment_path



    def grid_config_lists(self, *args): #configs_data, configs_architecture, configs_traininig):
        # here custom changes possible to make the configurations list creation suitable to needs
        # based on Stack/NTK set width / bottl, var to None

        # config_list_data, config_list_architecture, config_list_training = [], [], []
        # configs = configs_data.copy()
        # configs.update(configs_architecture)
        # configs.update(configs_traininig)

        configs = {}
        for config in args:
            configs.update(config)

        configs = self.dictvals_to_list(configs)

        grid = ParameterGrid(configs)

        args_list_of_configs = [[]] * len(args)

        for new_config in grid:

            for i in range(len(args)):

                temp_dict = {key: new_config[key] for key in args[i].keys()}
                args_list_of_configs[i] = args_list_of_configs[i] + [temp_dict]

            # temp_dict = {key: new_config[key] for key in configs_data.keys()}
            # config_list_data.append(temp_dict) 
            # temp_dict = {key: new_config[key] for key in configs_architecture.keys()}
            # config_list_architecture.append(temp_dict)
            # temp_dict = {key: new_config[key] for key in configs_traininig.keys()}
            # config_list_training.append(temp_dict) 
        
        # return config_list_data, config_list_architecture, config_list_training
        return args_list_of_configs