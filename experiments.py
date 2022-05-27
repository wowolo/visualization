import os
from pathlib import Path
import json

import util

class ExperimentManager():

    def __init__(self, nn_model, data_dict, root='.'):

        self.nn_model = nn_model

        self.x_train = data_dict['x_train']
        self.y_train = data_dict['y_train']
        self.x_val = data_dict['x_val']
        self.y_val = data_dict['y_val']

        self.root = Path(root)
        
        

    def do_exerimentbatch(self, list_of_complete_configs, experimentbatch_name):
        # might want to define complete configs at some point and check for them

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

        self.nn_model.save(experimentbatch_path / 'initial_nn_model_statedict.pt')

        experiment_index_dict = {'experiment_{}'.format(i): None for i in range(len(list_of_complete_configs))}

        for i, config in enumerate(list_of_complete_configs):
            # make training computations for given config
            self.nn_model.reset_parameters()
            self.nn_model.train(
                self.x_train, 
                self.y_train, 
                **config
            )

            # document experiment at experiment_path with: config files
            exp_ind = 'experiment_{}'.format(i)
            experiment_path = experimentbatch_path / exp_ind
            os.mkdir(experiment_path)
            

            # create add to index and write in experiment directory general config
            json_config = {key: util.make_jsonable(config[key]) for key in config.keys()}
            experiment_index_dict['experiment_{}'.format(i)] = json_config
            with open(experiment_path / 'general_experiment_config.txt', 'w') as file:
                file.write(json.dumps(json_config))

            # wite nn_model.config_architecture &  nn_model.config_training (partially updated after training)
            with open(experiment_path / 'nn_config_architecture.txt', 'w') as file:
                json_config_architecture = {key: util.make_jsonable(self.nn_model.config_architecture[key]) for key in self.nn_model.config_architecture.keys()}
                file.write(json.dumps(json_config_architecture))
            with open(experiment_path / 'nn_config_training.txt', 'w') as file:
                json_config_training = {key: util.make_jsonable(self.nn_model.config_training[key]) for key in self.nn_model.config_training.keys()}
                file.write(json.dumps(json_config_training))
            
            self.nn_model.save(experiment_path / 'nn_model_statedict.pt')

        # document experiment index file for whole batch
        with open(experimentbatch_path / 'experiment_index.txt', 'w') as file:
            file.write(json.dumps(experiment_index_dict))
