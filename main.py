import numpy as np
# import matplotlib.pyplot as plt

from create_data import CreateData


n_samples = 64
noise_scale = .5
x_min = -1
x_max = 1
n_val = 256
np.random.seed(seed=24)


# config 1
d_in = 2
d_out = 32
def f(x):
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

# config 2
# d_in = 1
# d_out = 1
# def f(x):
#     return np.sin(np.pi*x[:,0])

# config 3
# d_in = 1
# d_out = 2
# def f(x):
#     return np.stack([x[:,0]**2 -0.5, 2.0*(x[:,0]<0.3)*(x[:,0]-0.3)+1], axis=1)

data = CreateData(d_in, d_out, f)

# plt.plot(data.x_train, data.y_train, 'ko')
# plt.plot(data.x_val, data.y_val, 'g.')

# #np.stack([np.sin(np.pi*x_train[:,0]) +0.0, 2.0*(x_train[:,0]>0)-1], axis=1).shape
# data.y_train.shape

# plt.plot(data.x_train[:,0], data.y_train[:,0], 'g.')
# plt.plot(data.x_train[:,0], data.y_train[:,1], 'r.')