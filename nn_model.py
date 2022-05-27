import numpy as np
import torch
from torch import nn
import torch.optim as optim
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

    def init_arch_NTK(self, **kwargs):
        # extract necessary hyperparameters
        hyperparam = ['d_in', 'd_out', 'width', 'depth']
        hyperparam_dict = {key: kwargs[key] for key in hyperparam}

        mod_list = []
        depth = hyperparam_dict['depth']

        for i in range(depth):
            if depth == 1:
                mod_list.append(NTK_Linear(hyperparam_dict['d_in'], hyperparam_dict['d_out']))
            else:
                if i == 0:
                    mod_list.append(NTK_Linear(hyperparam_dict['d_in'], hyperparam_dict['width']))
                elif i < depth - 1:
                    mod_list.append(NTK_Linear(hyperparam_dict['width'], hyperparam_dict['width']))
                else:
                    mod_list.append(NTK_Linear(hyperparam_dict['width'], hyperparam_dict['d_out']))

        self.layers = nn.ModuleList(mod_list).double()

        return hyperparam_dict


    
    def forward_NTK(self, x):

        for layer in self.layers[:-1]:
            x = nn.ReLU()(layer(x))
        
        x = self.layers[-1](x)
        
        return x



    ########## Stack methods ##########

    def init_arch_Stack(self, **kwargs):
        # extract necessary hyperparameters
        hyperparam = ['d_in', 'd_out', 'bottleneck_width', 'variable_width', 'depth', 'skip_conn', 
            'linear_skip_conn', 'linear_skip_conn_width']
        hyperparam_dict = {key: kwargs[key] for key in hyperparam}

        mod_list = []
        depth = hyperparam_dict['depth']

        for i in range(depth):
            if depth == 1:
                mod_list.append(Stack_Core(
                            hyperparam_dict['d_in'], 
                            hyperparam_dict['d_out'],
                            hyperparam_dict['variable_width'],
                            hyperparam_dict['skip_conn'],
                            hyperparam_dict['linear_skip_conn'],
                            hyperparam_dict['linear_skip_conn_width']
                        )
                    )
            else:
                if i == 0:
                    mod_list.append(Stack_Core(
                            hyperparam_dict['d_in'], 
                            hyperparam_dict['bottleneck_width'],
                            hyperparam_dict['variable_width'],
                            hyperparam_dict['skip_conn'],
                            hyperparam_dict['linear_skip_conn'],
                            hyperparam_dict['linear_skip_conn_width']
                        )
                    )
                elif i < depth - 1:
                    mod_list.append(Stack_Core(
                            hyperparam_dict['bottleneck_width'], 
                            hyperparam_dict['bottleneck_width'],
                            hyperparam_dict['variable_width'],
                            hyperparam_dict['skip_conn'],
                            hyperparam_dict['linear_skip_conn'],
                            hyperparam_dict['linear_skip_conn_width']
                        )
                    )
                else:
                    mod_list.append(Stack_Core(
                            hyperparam_dict['bottleneck_width'], 
                            hyperparam_dict['d_out'],
                            hyperparam_dict['variable_width'],
                            hyperparam_dict['skip_conn'],
                            hyperparam_dict['linear_skip_conn'],
                            hyperparam_dict['linear_skip_conn_width']
                        )
                    )

        self.layers = nn.ModuleList(mod_list).double()

        return hyperparam_dict
    


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

        # parameter input by user
        dict_extraction_strings = [
            'architecture_key',
            'd_in', 
            'd_out', 
            'width', 
            'depth', 
            'bottleneck_width', 
            'variable_width', 
            'skip_conn',
            'linear_skip_conn',
            'linear_skip_conn_width'
        ]

        self.config_architecture = {string: kwargs[string] for string in dict_extraction_strings}        
        self.config_architecture['report'] = util.dict_extract(kwargs, 'report', True)

        self.config_training = {}

        hyperparam_dict = self.initialize_architecture()
        self.double()
        
        if self.config_architecture['report']: 
            nn_util.report_hyperparam(self.config_architecture['architecture_key'], hyperparam_dict)

    

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


# TODO implement tracking object to get history
    def train(self, x_train, y_train, **kwargs):

        if isinstance(self.config_architecture['architecture_key'], type(None)):
            return None

        # get parameters from kwargs
        dict_extraction_strings = [
            'criterion',
            'shuffle',
            'epochs',
            'batch_size',
            'regularization_alpha',
            'regularization_ord', 
            'learning_rate', 
            'update_rule'
        ]

        self.config_training = {string: kwargs[string] for string in dict_extraction_strings} 

        epochs = self.config_training['epochs']    
        
        # prepare torch objects needed in training loop
        optimizer = self.config_training['update_rule'](self.parameters(), lr=self.config_training['learning_rate'])
        training_generator = nn_util.DataGenerator(x_train, y_train, **self.config_training)

        self.loss_wout_reg = np.empty(epochs * training_generator.__len__())
        self.loss = np.empty_like(self.loss_wout_reg)
        ind_loss = 0

        for epoch in range(epochs):

            print('Epoch: ', epoch)

            for X, y in training_generator:

                output = self.forward(X)
                loss = self.config_training['criterion'](output, y)
                self.loss_wout_reg[ind_loss] = loss

                # add regularization terms to loss
                reg = torch.tensor(0., requires_grad=True)

                for param in self.parameters():
                    reg = reg + torch.linalg.vector_norm(param.flatten(), ord=self.config_training['regularization_ord'])**2
                
                loss = loss + self.config_training['regularization_alpha'] * reg
                self.loss[ind_loss] = loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ind_loss += 1



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