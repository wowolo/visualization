import torch


def report_config(config):

    print('The following configuration has been used for the construction of the {} \
architecture:'.format(config['architecture_key']))

    for paramkey in config.keys():
        print('{}: {}'.format(paramkey, config[paramkey]))



def create_config(kwargs, default_extraction_strings):

    config_architecture = {string: None for string in default_extraction_strings}

    for string in default_extraction_strings:
        
        if string in kwargs.keys():
            item = kwargs[string]
        else:
            item = default_extraction_strings[string]
        
        config_architecture[string] = item
    
    return config_architecture



class linear_activation(torch.nn.Module): # currently implemented as identity
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class dimred_MSELoss(torch.nn.Module):

    def __init__(self, dimension_activity):

        super(dimred_MSELoss,self).__init__()
        self.dimension_activity = dimension_activity
    

    
    def forward(self, output, target):

        if output.shape[0] == 0: return 0

        dimred_output = output[:, self.dimension_activity]
        dimred_target = target[:, self.dimension_activity]
        return torch.mean((dimred_output - dimred_target)**2)