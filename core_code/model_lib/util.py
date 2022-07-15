import torch


def report_config(config):

    print('The following configuration has been used for the construction of the {} \
architecture:'.format(config['architecture_key']))

    for paramkey in config.keys():
        print('{}: {}'.format(paramkey, config[paramkey]))



class identity_activation(torch.nn.Module): # currently implemented as identity
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


def _hidden_bottleneck_activation_fm(string):
    return {
        'Identity': identity_activation,
        'ReLU': torch.nn.ReLU,
    }[string]



def _hidden_layer_activation_fm(string):
    return {
        'Identity': identity_activation,
        'ReLU': torch.nn.ReLU,
    }[string]