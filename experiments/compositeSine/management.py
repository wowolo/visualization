import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from experiments.util import BasicManager
from core_code.create_data import CreateData
from core_code.nn_model import ExtendedModel



grayscale_list = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey', 'gainsboro']



class ExperimentManager(BasicManager):

    def __init__(self, configs_data, configs_architecture, configs_traininig, configs_custom, storage_path=None):

        self.set_randomness(configs_custom['torch_seed'], configs_custom['numpy_seed'])

        # save configs
        self.configs_data = configs_data 
        self.configs_architecture = configs_architecture
        self.configs_traininig = configs_traininig
        self.configs_custom = self.init_configs_custom(**configs_custom) # guarantee that it has the key determined by 'default_extraction_strings'

        # list of configs
        self.configs_data_list, self.configs_architecture_list, self.configs_training_list, self.configs_custom_list = self.grid_config_lists(configs_data, configs_architecture, configs_traininig, configs_custom)
    
        # create storage directory
        self.storage_path = self.create_storage_dir(storage_path, Path(__file__).parent)


    def init_configs_custom(self, **kwargs):

        default_extraction_strings = {
            'n_samples_per_loss': 256,
            'x_max_per_loss': 2,
            'save_fig': True,
            'torch_seed': 13,
            'numpy_seed': 33,
        }

        config = {string: None for string in default_extraction_strings}

        for string in default_extraction_strings:
            
            if string in kwargs.keys():
                item = kwargs[string]
            else:
                item = default_extraction_strings[string]
            
            config[string] = item
        
        return config



    def run(self, experimentbatch_name=None, storage_path=None):

        if isinstance(storage_path, type(None)):
            storage_path = self.storage_path

        # run for given self.configs: grid_config_lists
        experiment_num = len(self.configs_data_list)

        # might want to define complete configs at some point and check for them
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # send all tensors to device, i.e. data in traininig and model

        # write initial nn_model config params
        experimentbatch_path = self.default_experimentbatch_dir(storage_path, experimentbatch_name)
        self.write_experimentindex(experimentbatch_path)


        for i in range(experiment_num):

            # manage the structure configurations
            config_data = self.configs_data_list[i]
            config_architecture = self.configs_architecture_list[i]
            config_training = self.configs_training_list[i]
            config_custom = self.configs_custom_list[i]
            
            # initialize objects for given configs
            data = CreateData(
                **config_data
            )
            x_train, y_train = data.create_training_data()
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            nn_model = ExtendedModel(
                **config_architecture
            ).to(device)

            # create data and activity for each loss determined by config_custom
            num_losses = len(config_custom['n_samples_per_loss'])
            x_train_list = [[]] * num_losses
            y_train_list = [[]] * num_losses

            for j, (n_samples, x_max) in enumerate(zip(config_custom['n_samples_per_loss'], config_custom['x_max_per_loss'])):

                data.config['n_samples'] = n_samples
                data.config['x_max'] = x_max
                x_train_list[j], y_train_list[j] = data.create_training_data() 

            loss_activity = torch.cat([(j+1) * torch.ones(len(x_train_list[j])) for j in range(num_losses)])

            x_train = torch.cat(x_train_list, dim=0).to(device)
            y_train = torch.cat(y_train_list, dim=0).to(device)

            # might want to plot loss on valuation data as well? (needs to be implemented in training)
            x_val, y_val = data.create_valuation_data() 

            # sort x_val
            ind_sort = torch.argsort(x_val.detach(), axis=0)[:, 0]
            x_val = x_val[ind_sort]
            y_val = y_val[ind_sort]

            experiment_path = self.create_experiment_dir(experimentbatch_path, i) # numeric id i
            figure_path = experiment_path / 'figures' # needed for loss plot from training
            os.mkdir(figure_path)

            # train the model on the given data
            nn_model.train(
                x_train, 
                y_train,
                loss_activity, 
                x_val=x_val,
                y_val=y_val,
                figure_path=figure_path,
                **config_training
            )

            # create and save plots (unless turned off by config_custom)
            if config_custom['save_fig']:
                # x0min = float(min(x_train[:,0]) - 0.3)
                # x0max = float(max(x_train[:,0]) + 0.3)
                # x1min = float(min(x_train[:,1]) - 0.3)
                # x1max = float(max(x_train[:,1]) + 0.3)
                # nn_model.plot2d(x_train, y_train, x0min, x0max, x1min, x1max, dirname=experiment_path / 'figures')
                
                for i in range(config_data['d_out']):
                    plt.figure()
                    set_counter = 0
                    for loss_num, (_, active_dim) in enumerate(config_training['criterions']):
                        if i in active_dim:
                            plt.plot(x_train_list[loss_num], y_train_list[loss_num][:,i], 'o', color=grayscale_list[set_counter], markersize=3, label='Training data - Loss {}'.format(loss_num + 1))
                            set_counter = (set_counter + 1) % len(grayscale_list)
                            # if i == 0:
                            #     plt.plot(x_train_1, y_train_1[:,i], 'ko', markersize=8, label='Training data')
                            # else:
                            #     plt.plot(x_train_2, y_train_2[:,i], 'ko', markersize=8, label='Training data')
                    plt.plot(x_val.detach(), nn_model(x_val).detach()[:,i], 'r-', label='Stacked NN')
                    plt.plot(x_val.detach(), y_val.detach()[:,i], 'k-', label='True function')
                    plt.title('CompositeSine Experimen - Output dimension {}'.format(i))
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=2 + set_counter)
                    plt.savefig(figure_path / 'fig_{}.png'.format(i), bbox_inches="tight")
                    plt.close('all')
            
            # document the experiment within the experiment batch folder
            self.write_documentation(experiment_path, data, nn_model, config_custom)

    

    def write_documentation(self, experiment_path, data, nn_model, config_custom):
        # allocate complete documentation writing within this method, i.e. ... + create dict_content + save_network_weights
        dict_content = {}

        dict_content['config_data.txt'] = data.config
        dict_content['nn_config_architecture.txt'] = nn_model.config_architecture
        dict_content['nn_config_training.txt'] = nn_model.config_training
        dict_content['config_custom.txt'] = config_custom
        dict_content['loss_wout_reg.txt'] = nn_model.loss_wout_reg
        dict_content['loss.txt'] = nn_model.loss

        for filename, content in dict_content.items():

            if isinstance(content, dict):
                self.dict_to_file(content, experiment_path / filename)

            elif isinstance(content, list):
                self.list_to_file(content, experiment_path / filename)

            else:
                raise ValueError("The file '{}' is neither a dictionary nor a list and cannot be written to the documentation.".format(filename))

        # save network weights
        self.save_network_weights(experiment_path, nn_model)

    

    def write_experimentindex(self, experimentbatch_path):

        experiment_num = len(self.configs_data_list)

        experiment_index_dict = {}

        for i in range(experiment_num):

            experiment_path = self.create_experiment_dir(experimentbatch_path, i, print_only=True) # numeric id i

            # manage the structure configurations
            all_config = self.configs_data_list[i].copy()
            all_config.update(self.configs_architecture_list[i])
            all_config.update(self.configs_training_list[i])
            all_config.update(self.configs_custom_list[i])
            json_all_config = {key: self.make_jsonable(all_config[key]) for key in all_config.keys()}

            # update the experiment index
            experiment_index_dict[experiment_path.name] = json_all_config
        
        # document experiment index file for whole batch
        file_path = experimentbatch_path / 'experiment_index.txt'
        self.dict_to_file(experiment_index_dict, file_path)