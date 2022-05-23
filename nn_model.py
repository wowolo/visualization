import torch
from torch import nn

class CreateArchitecture(nn.Module): # the kwargs differ depending on the architecture: implement print function giving what has been used
    # in construction of architecture

    def __init__(self, key, d_in, d_out, **kwargs):

        super(CreateArchitecture, self).__init__()

        self.initialize_architecture(key)

        self.optimizer = torch.optim.Adam(self.neural_network.parameters()) # torch optim Adam of self.neural_network

    

    def initialize_architecture(self, key, **kwargs):

        architecture_dict = {
            'NTK': self.create_model_NTK,
            'Stack': self.create_model_stack}

        architecture_dict[key](**kwargs)

        return neural_network


    def create_model_NTK(self, **kwargs):
        # self initialization of weights 


    
    def create_model_stack(self, **kwargs):
        # self initialization of weights 


    def _core_stack(self, input_width, output_width, variable_width):

        stack = nn.Sequential(
            nn.Linear(input_width, variable_width),
            nn.ReLU(),
            nn.Linear(variable_width, output_width)
            )

        return stack

    

class NNModel(CreateArchitecture):

    def __init__(self):

        super(NNModel, self).__init__() # pass appropriate input to create architecture



    def forward(self, x):
        return self.neural_network(x)



    def train(self):



    def get_history(self): 



    def plot(self):
        # use functions from plot functions script