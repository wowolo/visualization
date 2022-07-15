from distutils.command.config import config
from core_code.util.helpers import create_config, check_config



def _make_init_config(default_extraction_strings, **kwargs):
    
    config_data = create_config(kwargs, default_extraction_strings)
    all_losses = check_config(config_data)
    
    return config_data, all_losses



def init_config_data(**kwargs):

    default_extraction_strings = {
        'd_in': None, 
        'd_out': None, 
        'f_true': None,
        'focus_ind': 0, 
        'n_train': 256, 
        'x_min_train': -1, 
        'x_max_train': 1, 
        'noise_scale': .1,
        'n_val': 128,
        'x_min_val': -1, 
        'x_max_val': 1, 
        'n_test': 128,
        'x_min_test': -1, 
        'x_max_test': 1, 
        'data_generators_train': 'equi',
        'data_generators_val': 'equi',
        'data_generators_test': 'equi',
    }
    
    return _make_init_config(default_extraction_strings, **kwargs)



def init_config_training(**kwargs):

    default_extraction_strings = {
        'criterion': 'MSELoss',
        'shuffle': True,
        'epochs': 1024, 
        'batch_size': 64,
        'regularization_alpha': 0.1, 
        'regularization_ord': 2,
        'learning_rate': 0.0001,
        'update_rule': 'Adam', 
        'data_loss_batching': True,
        'loss_plot': True,
    }

    return _make_init_config(default_extraction_strings, **kwargs)



def init_config_abcMLP(**kwargs):
    # extract necessary hyperparameters
    default_extraction_strings = {
        'architecture_key': 'abcMLP',
        'd_in': None, 
        'd_out': None, 
        'width': 1024, 
        'depth': 6, 
        'list_a': [-.5] + [0 for i in range(4)] + [.5], # default: mup
        'list_b': [.5 for i in range(6)], # default: mup
        'c': 0, # default: mup
        'hidden_layer_activation': 'ReLU',
    }
    
    return _make_init_config(default_extraction_strings, **kwargs)



def init_config_Stack(**kwargs):
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
    
    return _make_init_config(default_extraction_strings, **kwargs)