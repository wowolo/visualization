import numpy as np

# loss are enumerated starting with 0

config_function = {
    'd_in': 1, 
    'd_out': 7,
    'f_true': 'compositeSine',
    'focus_ind': 0
}

# configs for model architecture
configs_architecture = {
    'architecture_key': 'abcMLP',
    'depth': 6,
    'width': 4096, 
    'bottleneck_width': 4096, # for Stack
    'variable_width': 4096, # for Stack
    'linear_skip_conn': False, # for Stack
    'linear_skip_conn_width': 64, # for Stack
    'skip_conn': False, # for Stack
    'hidden_bottleneck_activation': 'Identity', 
    'hidden_layer_activation': 'ReLU', 
    # for abcMLP
    'list_a': [0] + [0.5 for i in range(5)], 
    'list_b': [0 for i in range(6)], 
    'c': 0, 
}
configs_architecture.update(config_function)

# configs for data creation
configs_data = {
    #### (potentially) loss specific ####
    'n_train': {'loss_0': 64, 'loss_1': 256},
    'noise_scale': .1,
    'x_min_train': -2,
    'x_max_train': {'loss_0': 0, 'loss_1': 2},
    'n_val': {'loss_0': 64, 'loss_1': 0},
    'x_min_val': 0,
    'x_max_val': 2,
    'n_test': {'loss_0': 64, 'loss_1': 256},
    'x_min_test': -2,
    'x_max_test': 2,
}
configs_data.update(config_function)

# configs for model training
configs_training = {
    'epochs': 64, #024, 
    'batch_size': 128, 
    'data_loss_batching': True,
    'learning_rate': 0.01,
    'update_rule': 'SGD',
    'regularization_alpha': 0.0,
    'regularization_ord': 2,
    #### (potentially) loss specific ####
    'criterion': {'loss_0': ('dimred_MSELoss', [0]), 'loss_1': ('dimred_MSELoss', list(np.arange(1, 7)))}, 
    'shuffle': True,
}

# configs customised for the specific experiment setup
configs_custom = {
    'torch_seed': 13,
    'numpy_seed': 33,
}