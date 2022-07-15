import torch
import core_code.util as util
import core_code.model_lib.util as nn_util

from core_code.util.default_config import init_config_Stack




class Stack_Core(torch.nn.Module):

    def __init__(self, input_width, output_width, variable_width, hidden_layer_activation, skip_conn, linear_skip_conn, linear_skip_conn_width):
        
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.variable_width = variable_width
        self.hidden_layer_activation = hidden_layer_activation

        self.skip_conn = skip_conn
        self.linear_skip_conn = linear_skip_conn
        self.linear_skip_conn_width = linear_skip_conn_width

        self.hidden_input_width = variable_width
        if self.skip_conn:
            self.hidden_input_width += input_width
        if self.linear_skip_conn:
            self.hidden_input_width += self.linear_skip_conn_width


        self.initialize_parameters()



    def initialize_parameters(self):

        self.linear_1 = torch.nn.Linear(self.input_width, self.variable_width)
        
        if self.linear_skip_conn:
            self.linear_skip = torch.nn.Linear(self.input_width, self.linear_skip_conn_width)

        self.linear_2 = torch.nn.Linear(self.hidden_input_width, self.output_width)
    


    def forward(self, x):

        activation = nn_util._hidden_layer_activation_fm(self.hidden_bottleneck_activation)
        hidden_input = activation()(self.linear_1(x))

        if self.skip_conn:
            hidden_input = torch.cat((hidden_input, x), 1)

        if self.linear_skip_conn:
            linear_input = self.linear_skip(x)
            hidden_input = torch.cat((hidden_input, linear_input), 1)

        y = self.linear_2(hidden_input)

        return y

    

    def reset_layer_parameters(self):
        self.initialize_parameters()




class NNModel(torch.nn.Module):    


    def __init__(self, **config_architecture):
        super(NNModel, self).__init__()

        self.config_architecture = init_config_Stack(**config_architecture)    
        self.layers = self.init_architecture()



    def init_config(self, **kwargs):
        # extract necessary hyperparameters
        default_extraction_strings = {
            'architecture_key': 'Stack',
            'd_in': None, 
            'd_out': None, 
            'bottleneck_width': 64, 
            'variable_width': 128, 
            'depth': 1, 
            'skip_conn': False, 
            'linear_skip_conn': False,
            'linear_skip_conn_width': 32,
            'hidden_bottleneck_activation': 'Identity',
            'hidden_bottleneck_activation': 'ReLU',
        }
        
        config_architecture = util.create_config(kwargs, default_extraction_strings)
        
        return config_architecture



    def init_architecture(self):
        mod_list = []
        depth = self.config_architecture['depth']

        for i in range(depth):
            if depth == 1:
                mod_list.append(Stack_Core(
                            self.config_architecture['d_in'], 
                            self.config_architecture['d_out'],
                            self.config_architecture['variable_width'],
                            self.config_architecture['hidden_layer_activation'],
                            self.config_architecture['skip_conn'],
                            self.config_architecture['linear_skip_conn'],
                            self.config_architecture['linear_skip_conn_width']
                        )
                    )
            else:
                if i == 0:
                    mod_list.append(Stack_Core(
                            self.config_architecture['d_in'], 
                            self.config_architecture['bottleneck_width'],
                            self.config_architecture['variable_width'],
                            self.config_architecture['hidden_layer_activation'],
                            self.config_architecture['skip_conn'],
                            self.config_architecture['linear_skip_conn'],
                            self.config_architecture['linear_skip_conn_width']
                        )
                    )
                elif i < depth - 1:
                    mod_list.append(Stack_Core(
                            self.config_architecture['bottleneck_width'], 
                            self.config_architecture['bottleneck_width'],
                            self.config_architecture['variable_width'],
                            self.config_architecture['hidden_layer_activation'],
                            self.config_architecture['skip_conn'],
                            self.config_architecture['linear_skip_conn'],
                            self.config_architecture['linear_skip_conn_width']
                        )
                    )
                else:
                    mod_list.append(Stack_Core(
                            self.config_architecture['bottleneck_width'], 
                            self.config_architecture['d_out'],
                            self.config_architecture['variable_width'],
                            self.config_architecture['hidden_layer_activation'],
                            self.config_architecture['skip_conn'],
                            self.config_architecture['linear_skip_conn'],
                            self.config_architecture['linear_skip_conn_width']
                        )
                    )

        layers = torch.nn.ModuleList(mod_list)

        return layers
    


    def forward(self, x):

        for layer in self.layers[:-1]:
            activation = nn_util._hidden_bottleneck_activation_fm(self.config_architecture['hidden_bottleneck_activation'])
            x = activation()(layer(x))
        
        x = self.layers[-1](x)
        
        return x