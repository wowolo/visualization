import torch
from torch import nn

import util
import nn_util
from custom_layers import Core_Stack, NTK_Linear

class ModelCatalogue(nn.Module): # the kwargs differ depending on the architecture: implement print function giving what has been used
    # in construction of architecture

    def __init__(self, key=None, **kwargs):

        super(ModelCatalogue, self).__init__()

        self.key = key
        self.d_in = util.dict_extract(kwargs, 'd_in', 1)
        self.d_out = util.dict_extract(kwargs, 'd_out', 1)

        self.init_arch_dict = {
            'NTK': self.init_arch_NTK,
            'Stack': self.init_arch_Stack
        }
        self.forward_dict = {
            'NTK': self.forward_NTK,
            'Stack': self.forward_Stack
        }
        self.train_dict = {
            'NTK': self.train_NTK,
            'Stack': self.train_Stack
        }

        self.initialize_architecture(**kwargs)

        # self.optimizer = torch.optim.Adam(self.neural_network.parameters()) # torch optim Adam of self.neural_network

    

    def initialize_architecture(self, **kwargs):
        if isinstance(self.key, type(None)):
            return None
        # kwargs: 'depth', 'width', 'l2reg'

        self.init_arch_dict[self.key](**kwargs) # creates paramters of architecture



    def forward(self, x):
        if isinstance(self.key, type(None)):
            return None
        # manage by dictionary
        return self.forward_dict[self.key](x)

    

    def train(self):
        if isinstance(self.key, type(None)):
            return None
        self.train_dict[self.key]()


    ########## NTK methods ##########

    def init_arch_NTK(self, **kwargs):

        # self initialization of weights 
        depth = util.dict_extract(kwargs, 'depth', 2)
        width = util.dict_extract(kwargs, 'width', 124)
        report = util.dict_extract(kwargs, 'report', True)

        module

        for i in range(depth):
            if i == 0:

        
        if report:
            nn_util.report_hyperparam(self.key, ) # input all the hyperparameters that have been used


    
    def forward_NTK(self, x):

    ########## Stack methods ##########

    def init_arch_Stack(self, **kwargs):
        # self initialization of weights 
    


   


    

class NNModel(ModelCatalogue):

    def __init__(self):

        super(NNModel, self).__init__() # pass appropriate input to create architecture



    def forward(self, x):
        return self.neural_network(x)



    def train(self):



    def get_history(self): 



    def plot(self):
        # use functions from plot functions script