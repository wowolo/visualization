from torch import nn

from core_code.nn_model.custom_layers import Stack_Core, NTK_Linear, abc_Layer
import core_code.util as util
import core_code.nn_model.util as nn_util


class ModelMethods(nn.Module):
    # parameters/layers all have to be contained in self.layers attribute
    # method names have to have form init_arch_{}, forward_{} (.format(key))

    @staticmethod
    def _hidden_bottleneck_activation_fm(string):
        return {
            'Identity': nn_util.identity_activation,
            'ReLU': nn.ReLU,
        }[string]
    
    @staticmethod
    def _hidden_layer_activation_fm(string):
        return {
            'Identity': nn_util.identity_activation,
            'ReLU': nn.ReLU,
        }[string]
            

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
            'hidden_layer_activation': 'ReLU',
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


    
    def forward_NTK(self, x, **config_architecture):

        for layer in self.layers[:-1]:
            activation = self._hidden_layer_activation_fm(config_architecture['hidden_layer_activation'])
            x = activation()(layer(x))
        
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
            'linear_skip_conn_width': 32,
            'hidden_bottleneck_activation': 'Identity',
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
    


    def forward_Stack(self, x, **config_architecture):

        for layer in self.layers[:-1]:
            activation = self._hidden_bottleneck_activation_fm(config_architecture['hidden_bottleneck_activation'])
            x = activation()(layer(x))
        
        x = self.layers[-1](x)
        
        return x



########## abcMLP methods ##########

    def init_config_abcMLP(self, **kwargs):
        # extract necessary hyperparameters
        default_extraction_strings = {
            'd_in': None, 
            'd_out': None, 
            'width': 1024, 
            'depth': 6, 
            'list_a': [-.5] + [0 for i in range(4)] + [.5], # default: mup
            'list_b': [.5 for i in range(6)], # default: mup
            'c': 0, # default: mup
            'hidden_layer_activation': 'ReLU',
        }
        
        config_architecture = nn_util.create_config(kwargs, default_extraction_strings)

        if isinstance(config_architecture['list_a'], int):
            config_architecture['list_a'] = [config_architecture['list_a'] for i in range(config_architecture['depth'])]
        if isinstance(config_architecture['list_b'], int):
            config_architecture['list_b'] = [config_architecture['list_b'] for i in range(config_architecture['depth'])]
        
        return config_architecture



    def init_arch_abcMLP(self, **config_architecture):
        # extract necessary hyperparameters
        mod_list = []
        d_in = config_architecture['d_in']
        d_out = config_architecture['d_out']
        depth = config_architecture['depth']
        width = config_architecture['width']
        list_a = config_architecture['list_a']
        list_b = config_architecture['list_b']

        for i in range(depth):
            if depth == 1:
                mod_list.append(abc_Layer(d_in, d_out, list_a[i], list_b[i]))
            else:
                if i == 0:
                    mod_list.append(abc_Layer(d_in, width, list_a[i], list_b[i]))
                elif i < depth - 1:
                    mod_list.append(abc_Layer(width, width, list_a[i], list_b[i]))
                else:
                    mod_list.append(abc_Layer(width, d_out, list_a[i], list_b[i]))

        self.layers = nn.ModuleList(mod_list).double()


    
    def forward_abcMLP(self, x, **config_architecture):

        for layer in self.layers[:-1]:
            activation = self._hidden_layer_activation_fm(config_architecture['hidden_layer_activation'])
            x = activation()(layer(x))
        
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
        self.device = None
        
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

        x = x.to(self.device)

        key = self.config_architecture['architecture_key']

        if isinstance(key, type(None)):
            return None

        method_keyword = 'forward_{}'.format(key)
        forward_method = getattr(self, method_keyword)

        return forward_method(x, **self.config_architecture)