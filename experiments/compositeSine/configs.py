from pathlib import Path
from datetime import datetime
import numpy as np
import torch
# import matplotlib.pyplot as plt

from create_data import CreateData
from nn_model import ExtendedModel
from experiments import ExperimentManager
import util
import nn_model.util as nn_util



# f_true functions
def f(x, focus_ind=0):
    return util.function_library('compositeSine')(x)

config_function = {
    'd_in': 1, # >= 1
    'd_out': 7,
    'f_true': f,
    'focus_ind': 0
}

# configs for data creation
configs_data = {
    # data parameters
    'n_samples': [256],
    'noise_scale': .1,
    'x_min': -2,
    'x_max': 2,
    'n_val': 512,
}
configs_data.update(config_function)

# configs for model architecture
configs_architecture = {
    # architecture parameters
    'architecture_key': 'Stack', # ['Stack', 'NTK'],
    'depth': 3, #], #,[1, 2, 6],
    'width': 512, #[16, 64, 256, 512, 2048, 8192], # for NTK
    'bottleneck_width': 256, # [16, 256, 512], # for Stack
    'variable_width': 1024, #[1024, 2048, 4096, 8192], #], #  [16, 256, 2048, 8192], # for Stack
    'linear_skip_conn': False, # for Stack
    'linear_skip_conn_width': 64, # for Stack
    'skip_conn': False, # for Stack
    'hidden_bottleneck_activation': nn_util.linear_activation, # for Stack
    'hidden_layer_activation': torch.nn.ReLU, # for NTK
}
configs_architecture.update(config_function)

# configs for model training
configs_traininig = {
    # training parameters
    'criterions': [[nn_util.dimred_MSELoss([0]), nn_util.dimred_MSELoss(np.arange(1, 7))]],
    'shuffle': True,
    'epochs': 2048, #[1024, 4096], # 4096,
    'batch_size': 64, #[64, 256],
    'regularization_alpha': 0.1, #[0.1, 0.01, 0],
    'regularization_ord': 2,
    'learning_rate': [0.0001],
    'update_rule': torch.optim.Adam, 
    'separate_loss_batching': True,
}