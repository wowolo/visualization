import os
from pathlib import Path
import json
from sklearn.model_selection import ParameterGrid
import torch

import util

class ExperimentManager():

    def __init__(self, ModelCreator, DataCreator, root='.'):

        # data and model creator based on configs
        self.DataCreator = DataCreator
        self.ModelCreator = ModelCreator

        self.root = Path(root)
        
        

    def do_exerimentbatch(self, configs_data_list, configs_architecture_list, configs_traininig_list, experimentbatch_name, save_fig=True):
        experiment_num = len(configs_data_list)

        # might want to define complete configs at some point and check for them
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # send all tensors to device, i.e. data in traininig and model

        # write initial nn_model config params
        experimentbatch_path = self.root / experimentbatch_name
        i = 0
        while True:
            try:
                os.mkdir(experimentbatch_path)
                break
            except FileExistsError:
                experimentbatch_path = self.root / (experimentbatch_name + '_{}'.format(i))
                i += 1
                continue

        experiment_index_dict = {'experiment_{}'.format(i): None for i in range(experiment_num)}

        for i in range(experiment_num):
            config_data = configs_data_list[i]
            config_architecture = configs_architecture_list[i]
            config_training = configs_traininig_list[i]

            # make training initialization and computations for given config
            data = self.DataCreator(
                **config_data
            )
            x_train = data.x_train.to(device)
            y_train = data.y_train.to(device)

            nn_model = self.ModelCreator(
                **config_architecture
            ).to(device)

            nn_model.train(
                x_train, 
                y_train, 
                **config_training
            )

            # document experiment at experiment_path with: config files
            exp_ind = 'experiment_{}'.format(i)
            experiment_path = experimentbatch_path / exp_ind
            os.mkdir(experiment_path)

            # create add to index and write in experiment directory general config
            config = config_data.copy()
            config.update(config_architecture)
            config.update(config_training)
            json_config = {key: util.make_jsonable(config[key]) for key in config.keys()}
            
            experiment_index_dict['experiment_{}'.format(i)] = json_config
            
            file_path = experiment_path / 'config_data.txt'
            util.dict_to_file(data.config, file_path)

            # write nn_model.config_architecture &  nn_model.config_training (partially updated after training)
            file_path = experiment_path / 'nn_config_architecture.txt'
            util.dict_to_file(nn_model.config_architecture, file_path)
            
            file_path = experiment_path / 'nn_config_training.txt'
            util.dict_to_file(nn_model.config_training, file_path)
            
            # write training loss without regularization term and loss 
            file_path = experiment_path / 'loss_wout_reg.txt'
            with open(file_path, 'w') as file:
                file.write(json.dumps(nn_model.loss_wout_reg))

            file_path = experiment_path / 'loss.txt'
            with open(file_path, 'w') as file:
                file.write(json.dumps(nn_model.loss))

            nn_model.save(experiment_path / 'nn_model_weights.pt')

            if save_fig:
                x0min = float(min(x_train[:,0]) - 0.3)
                x0max = float(max(x_train[:,0]) + 0.3)
                x1min = float(min(x_train[:,1]) - 0.3)
                x1max = float(max(x_train[:,1]) + 0.3)
                nn_model.plot2d(x_train, y_train, x0min, x0max, x1min, x1max, dirname=experiment_path / 'figures')

        # document experiment index file for whole batch
        file_path = experimentbatch_path / 'experiment_index.txt'
        util.dict_to_file(experiment_index_dict, file_path)



    def grid_config_lists(self, configs_data, configs_architecture, configs_traininig):
        # here custom changes possible to make the configurations list creation suitable to needs
        # based on Stack/NTK set width / bottl, var to None

        config_list_data, config_list_architecture, config_list_training = [], [], []
        configs = configs_data.copy()
        configs.update(configs_architecture)
        configs.update(configs_traininig)

        configs = util.dictvals_to_list(configs)

        grid = ParameterGrid(configs)

        for new_config in grid:
            temp_dict = {key: new_config[key] for key in configs_data.keys()}
            config_list_data.append(temp_dict) 
            temp_dict = {key: new_config[key] for key in configs_architecture.keys()}
            config_list_architecture.append(temp_dict)
            temp_dict = {key: new_config[key] for key in configs_traininig.keys()}
            config_list_training.append(temp_dict) 
        
        return config_list_data, config_list_architecture, config_list_training