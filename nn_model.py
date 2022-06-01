import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


import util
import nn_util
from custom_layers import Stack_Core, NTK_Linear




class ModelMethods(nn.Module):
    # parameters/layers all have to be contained in self.layers attribute
    # method names have to have form init_arch_{}, forward_{} (.format(key))

    def __init__(self):
        super(ModelMethods, self).__init__()



    ########## NTK methods ##########

    def init_config_NTK(self, **kwargs):
        # extract necessary hyperparameters
        default_extraction_strings = {
            'd_in': None, 
            'd_out': None, 
            'width': 64, 
            'depth': 3, 
        }
        
        config_architecture = nn_util.create_config(kwargs, default_extraction_strings)
        
        return config_architecture



    def init_arch_NTK(self, **config_architecture):
        # extract necessary hyperparameters
        mod_list = []
        d_in = config_architecture['d_in']
        d_out = config_architecture['d_out']
        depth = config_architecture['depth']
        width = config_architecture['width']

        for i in range(depth):
            if depth == 1:
                mod_list.append(NTK_Linear(d_in, d_out))
            else:
                if i == 0:
                    mod_list.append(NTK_Linear(d_in, width))
                elif i < depth - 1:
                    mod_list.append(NTK_Linear(width, width))
                else:
                    mod_list.append(NTK_Linear(width, d_out))

        self.layers = nn.ModuleList(mod_list).double()


    
    def forward_NTK(self, x):

        for layer in self.layers[:-1]:
            x = nn.ReLU()(layer(x))
        
        x = self.layers[-1](x)
        
        return x



    ########## Stack methods ##########

    def init_config_Stack(self, **kwargs):
        # extract necessary hyperparameters
        default_extraction_strings = {
            'd_in': None, 
            'd_out': None, 
            'bottleneck_width': 64, 
            'variable_width': 128, 
            'depth': 1, 
            'skip_conn': False, 
            'linear_skip_conn': False,
            'linear_skip_conn_width': 32
        }
        
        config_architecture = nn_util.create_config(kwargs, default_extraction_strings)
        
        return config_architecture



    def init_arch_Stack(self, **config_architecture):
        # extract necessary hyperparameters

        mod_list = []
        depth = config_architecture['depth']

        for i in range(depth):
            if depth == 1:
                mod_list.append(Stack_Core(
                            config_architecture['d_in'], 
                            config_architecture['d_out'],
                            config_architecture['variable_width'],
                            config_architecture['skip_conn'],
                            config_architecture['linear_skip_conn'],
                            config_architecture['linear_skip_conn_width']
                        )
                    )
            else:
                if i == 0:
                    mod_list.append(Stack_Core(
                            config_architecture['d_in'], 
                            config_architecture['bottleneck_width'],
                            config_architecture['variable_width'],
                            config_architecture['skip_conn'],
                            config_architecture['linear_skip_conn'],
                            config_architecture['linear_skip_conn_width']
                        )
                    )
                elif i < depth - 1:
                    mod_list.append(Stack_Core(
                            config_architecture['bottleneck_width'], 
                            config_architecture['bottleneck_width'],
                            config_architecture['variable_width'],
                            config_architecture['skip_conn'],
                            config_architecture['linear_skip_conn'],
                            config_architecture['linear_skip_conn_width']
                        )
                    )
                else:
                    mod_list.append(Stack_Core(
                            config_architecture['bottleneck_width'], 
                            config_architecture['d_out'],
                            config_architecture['variable_width'],
                            config_architecture['skip_conn'],
                            config_architecture['linear_skip_conn'],
                            config_architecture['linear_skip_conn_width']
                        )
                    )

        self.layers = nn.ModuleList(mod_list).double()
    


    def forward_Stack(self, x):

        for layer in self.layers[:-1]:
            x = nn.ReLU()(layer(x))
        
        x = self.layers[-1](x)
        
        return x



    ########## Generic class methods ##########

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.double()







class ModelCatalogue(ModelMethods): 

    def __init__(self, **kwargs):

        super(ModelCatalogue, self).__init__()

        self.config_architecture = self.initialize_config_architecture(**kwargs)        
        report = util.dict_extract(kwargs, 'report', True)

        self.config_training = {}

        self.initialize_architecture()
        self.double()
        
        if report: 
            nn_util.report_config(self.config_architecture)



    def initialize_config_architecture(self, **kwargs):
        
        key = util.dict_extract(kwargs, 'architecture_key', None)
        config_architecture = {'architecture_key': key}

        if isinstance(key, type(None)):
            return None

        method_keyword = 'init_config_{}'.format(key)
        init_method = getattr(self, method_keyword)
        
        config_architecture.update(init_method(**kwargs))
        
        return  config_architecture



    def initialize_architecture(self):

        key = self.config_architecture['architecture_key']

        if isinstance(key, type(None)):
            return None

        method_keyword = 'init_arch_{}'.format(key)
        init_method = getattr(self, method_keyword)
        
        return init_method(**self.config_architecture) # creates paramters of architecture



    def forward(self, x):

        key = self.config_architecture['architecture_key']

        if isinstance(key, type(None)):
            return None

        method_keyword = 'forward_{}'.format(key)
        forward_method = getattr(self, method_keyword)

        return forward_method(x)



   


    

class ExtendedModel(ModelCatalogue):

    def __init__(self, **kwargs):

        super(ExtendedModel, self).__init__(**kwargs) # pass appropriate input to create architecture

        self.loss = []
        self.loss_wout_reg = []



    def train(self, x_train, y_train, **kwargs):

        if isinstance(self.config_architecture['architecture_key'], type(None)):
            return None

        self.config_training = self.init_config_training(**kwargs)

        epochs = self.config_training['epochs']    
        
        # prepare torch objects needed in training loop
        optimizer = self.config_training['update_rule'](self.parameters(), lr=self.config_training['learning_rate'])
        training_generator = nn_util.DataGenerator(x_train, y_train, **self.config_training)

        self.loss_wout_reg = list(np.empty(epochs * training_generator.__len__()))
        self.loss = list(np.empty_like(self.loss_wout_reg))
        ind_loss = 0

        for epoch in range(epochs):

            print('Epoch: ', epoch)

            for X, y in training_generator:

                output = self.forward(X)
                loss = self.config_training['criterion'](output, y)
                self.loss_wout_reg[ind_loss] = float(loss)

                # add regularization terms to loss
                reg = torch.tensor(0., requires_grad=True)

                for param in self.parameters():
                    reg = reg + torch.linalg.vector_norm(param.flatten(), ord=self.config_training['regularization_ord'])**2
                
                loss = loss + self.config_training['regularization_alpha'] * reg
                self.loss[ind_loss] = float(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ind_loss += 1



    def init_config_training(self, **kwargs):

        default_extraction_strings = {
            'criterion': torch.nn.MSELoss(),
            'shuffle': True,
            'epochs': 1024, 
            'batch_size': 64,
            'regularization_alpha': 0.1, 
            'regularization_ord': 2,
            'learning_rate': 0.0001,
            'update_rule': torch.optim.Adam, 
        }

        config_training = nn_util.create_config(kwargs, default_extraction_strings)

        return config_training



    def plot1d(self, x, y, plot_xdim, plot_ydim):
        # use functions from plot functions script
        plt.plot(x[:,plot_xdim], y[:,plot_ydim], 'g.')
        plt.plot(x[:,plot_xdim], self.forward(x).detach()[:,plot_ydim], 'r.')
        plt.show()


    
    def save(self, path):
        torch.save(self.state_dict(), path)
    


    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()