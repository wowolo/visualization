# %%
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
def f_0(x, focus_ind=0):
    return np.array([
        1 * x[:,1]**2 - 0.4,
        #0.1 * x[:,1]**2,
        0.4 * np.sign(x[:,focus_ind]) * x[:,focus_ind]**2,
        np.exp(x[:,focus_ind]) - 1,
        x[:,focus_ind],
        x[:,focus_ind]**2 - 0.25,
        x[:,focus_ind]**3,
        -1 * np.exp(-x[:,focus_ind]) + 0.5,
        -3 * x[:,focus_ind],
        -3 * np.sign(x[:,focus_ind]) * x[:,focus_ind]**2,
        -3 * x[:,focus_ind]**3
        -0.5 * np.exp(x[:,focus_ind] - 1),
        -3 * (x[:,focus_ind] - 1),
        -3 * (x[:,focus_ind] - 1)**2 + 1.5,
        -3 * (x[:,focus_ind] - 1)**3,
        1 * np.exp(x[:,focus_ind] + 1) - 1,
        2 * (x[:,focus_ind] + 1),
        2 * np.sign(x[:,focus_ind]) * (x[:,focus_ind])**2,
        2 * (x[:,focus_ind] + 1)**3,
        0.2 * np.exp(x[:,focus_ind] + 0.5),
        0.2 * (x[:,focus_ind] + 0.5),
        0.2 * (x[:,focus_ind] + 0.5)**2,
        0.2 * (x[:,focus_ind] + 0.5)**3,
        -x[:,focus_ind]**2 + 0.5,
        x[:,focus_ind]**2,
        0.5 * np.sign(x[:,focus_ind]) * x[:,focus_ind]**2,
        2 * x[:,focus_ind]**3,
        4 * np.sign(x[:,focus_ind]) * x[:,focus_ind]**2,
        -np.sign(x[:,focus_ind]) * x[:,focus_ind]**2,
        8 * x[:,focus_ind],
        256 * np.sign(x[:,focus_ind]) * x[:,focus_ind]**2,
        1024 * x[:,focus_ind],
        64 * x[:,focus_ind]**3,
        -100 * np.sign(x[:,focus_ind]) * x[:,focus_ind]**2
    ]).T



def f_1(x, focus_ind=0):
    return np.array([
        np.sin(np.pi*x[:,focus_ind])
    ]).T



def f_2(x, focus_ind=0):
    return np.stack([x[:,focus_ind]**2 -0.5, 2.0*(x[:,focus_ind]<0.3)*(x[:,focus_ind]-0.3)+1], axis=1)



def f_3(x, focus_ind=0):
    return util.function_library('compositeSine')(x)

# defer config to bash script to run code?

config_0 = {
    'd_in': 2, #[2, 8, 24], # >= 2
    'd_out': 32,
    'f_true': f_0,
    'focus_ind': 0
}

config_1 = {
    'd_in': 1,
    'd_out': 1,
    'f_true': f_1,
    'focus_ind': 0
}

config_2 = {
    'd_in': 1, # >= 2
    'd_out': 2,
    'f_true': f_2,
    'focus_ind': 0
}

config_3 = {
    'd_in': 1, # >= 1
    'd_out': 7,
    'f_true': f_3,
    'focus_ind': 0
}

config_function = config_3

# loss_ratio = util.create_loss_ratio(config_function['d_out'])

# configs file
configs_data = {
    # data parameters
    'n_samples': [256],
    'noise_scale': .1,
    'x_min': -2,
    'x_max': 2,
    'n_val': 512,
}
configs_data.update(config_function)

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




# %% cell 1
# TODO create new script handling data creation, model initialization and training + loss activity 
# or “new” ExperimentManager
# TODO integrate 1d plotting into extended model and the ExperimentManager (automatic recognition/keyword)
np.random.seed(seed=24)
manager = ExperimentManager(ExtendedModel, CreateData)
configs_data_list, configs_architecture_list, configs_traininig_list = manager.grid_config_lists(
    configs_data, 
    configs_architecture, 
    configs_traininig
)
timestamp = datetime.now().strftime('%H-%M_%d.%m.%Y')
# manager.do_exerimentbatch(configs_data_list, configs_architecture_list, configs_traininig_list, 'experiments_{}'.format(timestamp), save_fig=False)
# %% cell 2
# replace the cell above by the ExperimentManager (allowing for robust documentation?!)
# data_dict = {
#     'x_train': x_train,
#     'y_train': y_train,
#     'x_val': x_val,
#     'y_val': y_val
# }
# config_ = configs
# config_['epochs'] = 5
# config_['epochs'] = 1
# manager = ExperimentManager(nn_model, data_dict)
# manager.do_exerimentbatch([configs, config_], 'test_experiment')

# %% cell 3
# create data and model
data = CreateData(
    **configs_data_list[0]
)

nn_model = ExtendedModel(
    **configs_architecture_list[0]
)

# set all involved tensors to device gpu/cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # send all tensors to device, i.e. data in traininig and model

# create 64 samples for loss 1 and 256 samples for loss 2
n_samples_1 = 256
x_max_1 = [0]
data.config['n_samples'] = n_samples_1
data.config['x_max'] = x_max_1
y_train_1, x_train_1 = data.create_training_data() 

n_samples_2 = 0
x_max_2 = [2]
data.config['n_samples'] = n_samples_2
data.config['x_max'] = x_max_2
y_train_2, x_train_2 = data.create_training_data() 

x_train = torch.cat((x_train_1, x_train_2), dim=0).to(device)
y_train = torch.cat((y_train_1, y_train_2), dim=0).to(device)


x_val = data.x_val # .to(device)
y_val = data.y_val # .to(device)
nn_model.to(device)

# sort x_val
ind_sort = torch.argsort(x_val.detach(), axis=0)[:, 0]
x_val = x_val[ind_sort]
y_val = y_val[ind_sort]

# train the model on the given data
loss_activity = torch.cat((1 * torch.ones(n_samples_1) , 2 * torch.ones(n_samples_2)))
nn_model.train(
    x_train, 
    y_train,
    loss_activity, 
    **configs_traininig_list[0]
)

# nn_model.plot2d(x_train, y_train, -1 , 1, -3, 3, save=True, dirname=Path().cwd() / 'figures')


# # evaluate the trained model by plots
# plt.plot(data.x_val[:,1], data.y_val[:,0], 'g.')
# plt.plot(data.x_val[:,1], nn_model.forward(data.x_val).detach()[:,0], 'r.')
# plt.show()

# for targ_func in range(1, 32):
#     plt.plot(data.x_val[:,0], data.y_val[:,targ_func], 'g.')
#     plt.plot(data.x_val[:,0], nn_model.forward(data.x_val).detach()[:,targ_func], 'r.')
#     plt.show()
# %% cell 4
# loading model - previously need to initialize model in cell 3 
# path = Path().cwd() / 'experiments_13-50_19.06.2022' / 'experiment_0' / 'nn_model_weights.pt'
# path = Path().cwd() / 'experiments_14-33_19.06.2022' / 'experiment_0' / 'nn_model_weights.pt'

# nn_model.load_state_dict(torch.load(path))

import matplotlib.pyplot as plt
fig_folder = Path().cwd() / 'exp_1'
for i in range(7):
    plt.figure()
    if i == 0:
        plt.plot(x_train_1, y_train_1[:,i], 'ko', markersize=8, label='Training data')
    else:
        plt.plot(x_train_2, y_train_2[:,i], 'ko', markersize=8, label='Training data')
    plt.plot(x_val.detach(), nn_model(x_val).detach()[:,i], 'r-', label='Stacked NN')
    plt.plot(x_val.detach(), y_val.detach()[:,i], 'k-', label='True function')
    plt.title('Experiment 1 - Loss 1.')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3)
    plt.savefig(fig_folder / 'temp_fig_{}.png'.format(i), bbox_inches="tight")
# %%
# Stack: depth 3, widths 256/1024
# NTK:  depth 6, width 
# %%
