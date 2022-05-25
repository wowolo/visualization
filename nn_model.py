from tkinter import Y
import torch
from torch import nn
import torch.optim as optim

import util
import nn_util
from custom_layers import Stack_Core, NTK_Linear






class ModelMethods(nn.Module):
    # method names have to have form init_arch_{}, forward_{} (.format(key))

    def __init__(self):
        super(ModelMethods, self).__init__()



    ########## NTK methods ##########

    def init_arch_NTK(self, **kwargs):
        # extract necessary hyperparameters
        hyperparam = ['d_in', 'd_out', 'width', 'depth']
        hyperparam_dict = {key: kwargs[key] for key in hyperparam}

        mod_list = []

        for i in range(hyperparam_dict['depth']):
            if i == 0:
                mod_list.append(NTK_Linear(hyperparam_dict['d_in'], hyperparam_dict['width']))
            elif i < hyperparam_dict['depth'] - 1:
                mod_list.append(NTK_Linear(hyperparam_dict['width'], hyperparam_dict['width']))
            else:
                mod_list.append(NTK_Linear(hyperparam_dict['width'], hyperparam_dict['d_out']))

        self.layers = nn.ModuleList(mod_list)

        return hyperparam_dict


    
    def forward_NTK(self, x):

        for layer in self.layers:
            y = layer(x)
        
        return y



    ########## Stack methods ##########

    def init_arch_Stack(self, **kwargs):
        # extract necessary hyperparameters
        hyperparam = ['d_in', 'd_out', 'bottleneck_width', 'variable_width', 'depth', 'skip_conn', 
            'linear_skip_conn', 'linear_skip_conn_width']
        hyperparam_dict = {key: kwargs[key] for key in hyperparam}

        mod_list = []

        for i in range(hyperparam_dict['depth']):
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
            elif i < hyperparam_dict['depth'] - 1:
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

        self.layers = nn.ModuleList(mod_list)

        return hyperparam_dict
    


    def forward_Stack(self, x):

        for layer in self.layers:
            y = layer(x)
        
        return y










class ModelCatalogue(ModelMethods): # the kwargs differ depending on the architecture: implement print function giving what has been used
    # in construction of architecture

    def __init__(self, key=None, **kwargs):

        super(ModelCatalogue, self).__init__()

        # parameter input by user
        self.config_params = kwargs
        self.config_params['key'] = key

        # adding default parameters if not given at initialization
        self.config_params['d_in'] = util.dict_extract(kwargs, 'd_in', 1)
        self.config_params['d_out'] = util.dict_extract(kwargs, 'd_out', 1)
        self.config_params['width'] = util.dict_extract(kwargs, 'width', 15)
        self.config_params['depth'] = util.dict_extract(kwargs, 'depth', 1)
        self.config_params['bottleneck_width'] = util.dict_extract(kwargs, 'bottleneck_width', 8)
        self.config_params['variable_width'] = util.dict_extract(kwargs, 'variable_width', 24)
        self.config_params['skip_conn'] = util.dict_extract(kwargs, 'skip_conn', False)
        self.config_params['linear_skip_conn'] = util.dict_extract(kwargs, 'linear_skip_conn', False)
        self.config_params['linear_skip_conn_width'] = util.dict_extract(kwargs, 'linear_skip_conn_width', 1)
        self.config_params['report'] = util.dict_extract(kwargs, 'report', True)

        hyperparam_dict = self.initialize_architecture()
        
        if self.config_params['report']: 
            nn_util.report_hyperparam(self.config_params['key'], hyperparam_dict)

        # self.optimizer = torch.optim.Adam(self.neural_network.parameters()) # torch optim Adam of self.neural_network

    

    def initialize_architecture(self):

        key = self.config_params['key']

        if isinstance(key, type(None)):
            return None

        method_keyword = 'init_arch_{}'.format(key)
        init_method = getattr(self, method_keyword)
        
        return init_method(**self.config_params) # creates paramters of architecture



    def forward(self, x):

        key = self.config_params['key']

        if isinstance(key, type(None)):
            return None
        method_keyword = 'forward_{}'.format(key)
        forward_method = getattr(self, method_keyword)

        return forward_method(x)



   


    

class ExtendedModel(ModelCatalogue):

    def __init__(self, key, **kwargs):

        super(ExtendedModel, self).__init__(key, **kwargs) # pass appropriate input to create architecture


# TODO implement tracking object to get history
    def train(self, x_train, y_train, criterion, **kwargs):

        if isinstance(self.config_params['key'], type(None)):
            return None

        # get parameters from kwargs
        epochs = util.dict_extract(kwargs, 'epochs', 1)
        learning_rate = util.dict_extract(kwargs, 'learning_rate', .001)
        update_rule = util.dict_extract(kwargs, 'optimizer', optim.SGD)
        regularization_ord = util.dict_extract(kwargs, 'regularization_ord', 2)
        regularization_alpha = util.dict_extract(kwargs, 'regularization_alpha', .005)
        
        # prepare torch objects needed in training loop
        optimizer = update_rule(self.parameters(), lr=learning_rate)
        training_generator = nn_util.DataGenerator(x_train, y_train, **kwargs)

        for epoch in range(epochs):

            print('Epoch: ', epoch)

            for X, y in training_generator:

                output = self.forward(X)
                loss = criterion(output, y)

                # add regularization terms to loss
                reg = torch.tensor(0., requires_grad=True)

                for param in self.parameters():
                    reg = reg + torch.linalg.vector_norm(param.flatten(), ord=regularization_ord)
                
                loss = loss + regularization_alpha * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


#     def get_history(self): 



#     def plot(self):
#         # use functions from plot functions script