import json
import numpy as np
from sklearn.model_selection import ParameterGrid



class BasicManager():

    
    def grid_config_lists(self, configs_data, configs_architecture, configs_traininig):
        # here custom changes possible to make the configurations list creation suitable to needs
        # based on Stack/NTK set width / bottl, var to None

        config_list_data, config_list_architecture, config_list_training = [], [], []
        configs = configs_data.copy()
        configs.update(configs_architecture)
        configs.update(configs_traininig)

        configs = self.dictvals_to_list(configs)

        grid = ParameterGrid(configs)

        for new_config in grid:
            temp_dict = {key: new_config[key] for key in configs_data.keys()}
            config_list_data.append(temp_dict) 
            temp_dict = {key: new_config[key] for key in configs_architecture.keys()}
            config_list_architecture.append(temp_dict)
            temp_dict = {key: new_config[key] for key in configs_traininig.keys()}
            config_list_training.append(temp_dict) 
        
        return config_list_data, config_list_architecture, config_list_training



    @staticmethod
    def make_jsonable(x):
        try:
            json.dumps(x)
            return x
        except:
            return str(x)



    @staticmethod
    def dict_to_file(dict, file_path, format='v'):
        # format: 'v' or 'h'
        with open(file_path, 'w') as file:
            if format == 'v':
                for key, val in dict.items():
                    file.write('{}: {}\n'.format(key, val))
            else:
                json_dict = {key: make_jsonable(dict[key]) for key in dict.keys()}
                file.write(json.dumps(json_dict))



    @staticmethod
    def dictvals_to_list(dict):
        for key, val in dict.items():
            if not(isinstance(val, list)):
                dict[key] = [val] 

        return dict