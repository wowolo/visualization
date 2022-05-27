# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

from create_data import CreateData
from nn_model import ExtendedModel
from experiments import ExperimentManager



# f_true functions
def f_0(x):
    return np.array([
        1 * x[:,1]**2 - 0.4,
        #0.1 * x[:,1]**2,
        0.4 * np.sign(x[:,0]) * x[:,0]**2,
        np.exp(x[:,0]) - 1,
        x[:,0],
        x[:,0]**2 - 0.25,
        x[:,0]**3,
        -1 * np.exp(-x[:,0]) + 0.5,
        -3 * x[:,0],
        -3 * np.sign(x[:,0]) * x[:,0]**2,
        -3 * x[:,0]**3
        -0.5 * np.exp(x[:,0] - 1),
        -3 * (x[:,0] - 1),
        -3 * (x[:,0] - 1)**2 + 1.5,
        -3 * (x[:,0] - 1)**3,
        1 * np.exp(x[:,0] + 1) - 1,
        2 * (x[:,0] + 1),
        2 * np.sign(x[:,0]) * (x[:,0])**2,
        2 * (x[:,0] + 1)**3,
        0.2 * np.exp(x[:,0] + 0.5),
        0.2 * (x[:,0] + 0.5),
        0.2 * (x[:,0] + 0.5)**2,
        0.2 * (x[:,0] + 0.5)**3,
        -x[:,0]**2 + 0.5,
        x[:,0]**2,
        0.5 * np.sign(x[:,0]) * x[:,0]**2,
        2 * x[:,0]**3,
        4 * np.sign(x[:,0]) * x[:,0]**2,
        -np.sign(x[:,0]) * x[:,0]**2,
        8 * x[:,0],
        256 * np.sign(x[:,0]) * x[:,0]**2,
        1024 * x[:,0],
        64 * x[:,0]**3,
        -100 * np.sign(x[:,0]) * x[:,0]**2
    ]).T



def f_1(x):
    return np.array([
        np.sin(np.pi*x[:,0])
    ]).T



def f_2(x):
    return np.stack([x[:,0]**2 -0.5, 2.0*(x[:,0]<0.3)*(x[:,0]-0.3)+1], axis=1)



config_0 = {
    'd_in': 8, # >= 2
    'd_out': 32,
    'f_true': f_0
}

config_1 = {
    'd_in': 1,
    'd_out': 1,
    'f_true': f_1
}

config_2 = {
    'd_in': 1, # >= 2
    'd_out': 2,
    'f_true': f_2
}

# config file
config = {
    # data parameters
    'n_samples': 256,
    'noise_scale': .1,
    'x_min': -1,
    'x_max': 1,
    'n_val': 128,
    # architecture parameters
    'architecture_key': 'Stack', # 'NTK'
    'depth': 2,
    'width': 1024,
    'bottleneck_width': 128, # for Stack
    'variable_width': 8192, # for Stack
    'linear_skip_conn': False, # for Stack
    'linear_skip_conn_width': 64, # for Stack
    'skip_conn': True, # for Stack
    # training parameters
    'criterion': torch.nn.MSELoss(),
    'shuffle': True,
    'epochs': 1024,
    'batch_size': 128,
    'regularization_alpha': 0.05,
    'regularization_ord': 2,
    'learning_rate': 0.0001,
    'update_rule': torch.optim.Adam, 
}
# add function specific parameters based on the configurations above
config.update(config_0)

np.random.seed(seed=24)

# %%

# create data and model
data = CreateData(
    **config
)

nn_model = ExtendedModel(
    **config
)

# set all involved tensors to device gpu/cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # send all tensors to device, i.e. data in traininig and model

x_train = data.x_train.to(device)
y_train = data.y_train.to(device)
x_val = data.x_val # .to(device)
y_val = data.y_val # .to(device)
nn_model.to(device)

# %%
# # train the model on the given data
# nn_model.train(
#     x_train, 
#     y_train, 
#     **config
# )

# # evaluate the trained model by plots
# plt.plot(data.x_val[:,1], data.y_val[:,0], 'g.')
# plt.plot(data.x_val[:,1], nn_model.forward(data.x_val).detach()[:,0], 'r.')
# plt.show()

# for targ_func in range(1, 32):
#     plt.plot(data.x_val[:,0], data.y_val[:,targ_func], 'g.')
#     plt.plot(data.x_val[:,0], nn_model.forward(data.x_val).detach()[:,targ_func], 'r.')
#     plt.show()

# %%
# replace the cell above by the ExperimentManager (allowing for robust documentation?!)
data_dict = {
    'x_train': x_train,
    'y_train': y_train,
    'x_val': x_val,
    'y_val': y_val
}
config_ = config
config_['epochs'] = 5
config_['epochs'] = 1
manager = ExperimentManager(nn_model, data_dict)
manager.do_exerimentbatch([config, config_], 'test_experiment')
# %%
