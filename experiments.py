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
        
        

    def do_exerimentbatch(self, list_of_complete_configs, experimentbatch_name):
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

        experiment_index_dict = {'experiment_{}'.format(i): None for i in range(len(list_of_complete_configs))}

        for i, config in enumerate(list_of_complete_configs):

            # make training initialization and computations for given config
            self.nn_model = self.ModelCreator(
                **config
            ).to(device)

            data = self.DataCreator(
                **config
            )
            x_train = data.x_train.to(device)
            y_train = data.y_train.to(device)

            self.nn_model.train(
                x_train, 
                y_train, 
                **config
            )

            # document experiment at experiment_path with: config files
            exp_ind = 'experiment_{}'.format(i)
            experiment_path = experimentbatch_path / exp_ind
            os.mkdir(experiment_path)

            # create add to index and write in experiment directory general config
            json_config = {key: util.make_jsonable(config[key]) for key in config.keys()}
            
            experiment_index_dict['experiment_{}'.format(i)] = json_config
            
            file_path = experiment_path / 'general_experiment_config.txt'
            util.dict_to_file(json_config, file_path)

            # write nn_model.config_architecture &  nn_model.config_training (partially updated after training)
            file_path = experiment_path / 'nn_config_architecture.txt'
            util.dict_to_file(self.nn_model.config_architecture, file_path)
            
            file_path = experiment_path / 'nn_config_training.txt'
            util.dict_to_file(self.nn_model.config_training, file_path)
            
            # write training loss without regularization term and loss 
            file_path = experiment_path / 'loss_wout_reg.txt'
            with open(file_path, 'w') as file:
                file.write(json.dumps(self.nn_model.loss_wout_reg))

            file_path = experiment_path / 'loss.txt'
            with open(file_path, 'w') as file:
                file.write(json.dumps(self.nn_model.loss))

            self.nn_model.save(experiment_path / 'nn_model_statedict.pt')

        # document experiment index file for whole batch
        file_path = experimentbatch_path / 'experiment_index.txt'
        util.dict_to_file(experiment_index_dict, file_path)



    def create_config_list(self, configs):
        # here custom changes possible to make the configurations list creation suitable to needs
        # based on Stack/NTK set width / bottl, var to None

        list_of_complete_configs = []  
        configs = util.dictvals_to_list(configs)
        grid = ParameterGrid(configs)

        for new_config in grid:
        
            if new_config['architecture_key'] == 'Stack':
                # set Stack non-relevant architecture parameters to None
                configs['width'] = None
            else:
                # set Stack relevant architecture parameters to None
                configs['bottleneck_width'] = None 
                configs['variable_width'] = None 
                configs['linear_skip_conn'] = None 
                configs['linear_skip_conn_width'] = None 
                configs['skip_conn'] = None

            list_of_complete_configs.append(new_config)  
        
        return list_of_complete_configs