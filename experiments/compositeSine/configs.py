import numpy as np

config_function = {
    'd_in': 1, # >= 1
    'd_out': 7,
    'f_true': 'compositeSine',
    'focus_ind': 0
}

# configs for data creation
configs_data = {
    # data parameters
    'n_samples': [256],
    'noise_scale': .1,
    'x_min': -2,
    'x_max': 2,
    'n_val': 256,
}
configs_data.update(config_function)

# configs for model architecture
configs_architecture = {
    'architecture_key': ['abcMLP'],
    'depth': 6,
    'width': 4096, 
    'bottleneck_width': 4096, # [16, 256, 512], # for Stack
    'variable_width': 4096, #[1024, 2048, 4096, 8192], #], #  [16, 256, 2048, 8192], # for Stack
    'linear_skip_conn': False, # for Stack
    'linear_skip_conn_width': 64, # for Stack
    'skip_conn': False, # for Stack
    'hidden_bottleneck_activation': 'Identity', #nn_util.identity_activation, # for Stack
    'hidden_layer_activation': 'ReLU', #torch.nn.ReLU, # for NTK
    # for abcMLP
    'list_a': [[0] + [0.5 for i in range(5)]], 
    'list_b': [[0 for i in range(6)]], 
    'c': 0, 
}
configs_architecture.update(config_function)

# configs for model training
configs_traininig = {
    'criterions': 'MSELoss', # [[('dimred_MSELoss', [0]),('dimred_MSELoss', list(np.arange(1, 7)))]], 
    'shuffle': True,
    'epochs': 1024, 
    'batch_size': [1, 8, 16, 32], 
    'regularization_alpha': 0.0,
    'regularization_ord': 2,
    'learning_rate': [0.1, 0.05, 0.01],
    'update_rule': ['SGD'],
    'separate_loss_batching': True,
}

# configs customised for the specific experiment setup
configs_custom = {
    'n_samples_per_loss': [[256]],
    'x_max_per_loss': [[2]],
    'save_fig': True,
    'torch_seed': 13,
    'numpy_seed': 33,
}