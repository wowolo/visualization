import numpy as np
import torch
# import matplotlib.pyplot as plt

import core_code.util as util
import core_code.nn_model.util as nn_util



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
    'n_val': 64 + 256,
}
configs_data.update(config_function)

# configs for model architecture
configs_architecture = {
    # architecture parameters
    'architecture_key': ['Stack'],
    'depth': 3,
    'width': 4096, # for NTK
    'bottleneck_width': 4096, # [16, 256, 512], # for Stack
    'variable_width': 4096, #[1024, 2048, 4096, 8192], #], #  [16, 256, 2048, 8192], # for Stack
    'linear_skip_conn': False, # for Stack
    'linear_skip_conn_width': 64, # for Stack
    'skip_conn': False, # for Stack
    #########
    'hidden_bottleneck_activation': 'Identity', #nn_util.identity_activation, # for Stack
    #########
    'hidden_layer_activation': 'ReLU', #torch.nn.ReLU, # for NTK
    # for abcMLP
    'list_a': [[-.5] + [0 for i in range(4)] + [.5]], # default: mup
    'list_b': [[.5 for i in range(6)]], # default: mup
    'c': 0, # default: mup
}
configs_architecture.update(config_function)

# configs for model training
configs_traininig = {
    # training parameters
    #########
    'criterions': [[('dimred_MSELoss', [0]),('dimred_MSELoss', list(np.arange(1, 7)))]], #[[nn_util.dimred_MSELoss([0]), nn_util.dimred_MSELoss(np.arange(1, 7))]],
    'shuffle': True,
    'epochs': 1024, #2048, #[1024, 4096], # 4096,
    'batch_size': 64, #[64, 256],
    'regularization_alpha': 0.0005,
    'regularization_ord': 2,
    'learning_rate': [0.001],
    #########
    'update_rule': 'SGD', #torch.optim.Adam, 
    'separate_loss_batching': True,
}

# configs customised for the specific experiment setup
configs_custom = {
    'n_samples_per_loss': [[64, 256]],
    'x_max_per_loss': [[0, 2]],
    'save_fig': True,
    'torch_seed': 13,
    'numpy_seed': 33,
}