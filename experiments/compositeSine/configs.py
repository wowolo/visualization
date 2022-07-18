import numpy as np

# task are enumerated starting with 0

config_function = {
    'd_in': 1, # one d_in value across all tasks, due to NN architecture
    'd_out': 7, # one d_out value across all tasks, due to NN architecture
    'f_true': 'compositeSine',
}

# configs for model architecture
configs_architecture = {
    'architecture_key': 'Stack',
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
    'list_a': [[0] + [0.5 for i in range(5)]], 
    'list_b': [[0 for i in range(6)]], 
    'c': 0, 
}
configs_architecture.update(config_function)

# configs for data creation
configs_data = {
    #### (potentially) task specific ####
    'n_train': {'task_0': 128, 'task_1': 512},
    'noise_scale': .1,
    'x_min_train': -2,
    'x_max_train': {'task_0': 0, 'task_1': 2},
    'n_val': {'task_0': 512, 'task_1': 512},
    'x_min_val': -2,
    'x_max_val': 2,
    'n_test': {'task_0': 64, 'task_1': 256},
    'x_min_test': -2,
    'x_max_test': 2,
}
configs_data.update(config_function)

# configs for model training
configs_training = {
    'epochs': 512, #024, 
    'batch_size': 64, 
    'data_task_batching': True,
    'learning_rate': 0.01,
    'update_rule': 'SGD',
    'regularization_alpha': 0.0001,
    'regularization_ord': 2,
    #### (potentially) task specific ####
    'criterion': {'task_0': ('dimred_MSELoss', [0]), 'task_1': ('dimred_MSELoss', list(np.arange(1, 7)))}, 
    'shuffle': True,
}

# configs trainer from bash file, else from default config

# configs customised for the specific experiment setup
configs_custom = {
    'logger': True,
    'callbacks': True,
    'seed': 77,
    'workers': True
}