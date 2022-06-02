import os
from pathlib import Path
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib import ticker, colors
import matplotlib.pyplot as plt

from nn_model.model_catalogue import ModelCatalogue
import nn_model.util as nn_util


    

class ExtendedModel(ModelCatalogue):
    # adds a training and various plotting methods

    def __init__(self, **kwargs):

        super(ExtendedModel, self).__init__(**kwargs) # pass appropriate input to create architecture

        self.loss = []
        self.loss_wout_reg = []



    def train(self, x_train, y_train, **kwargs):

        if isinstance(self.config_architecture['architecture_key'], type(None)):
            return None

        self.config_training = self.init_config_training(**kwargs)

        epochs = self.config_training['epochs']    
        
        # prepare torch objects needed in training loop
        optimizer = self.config_training['update_rule'](self.parameters(), lr=self.config_training['learning_rate'])
        training_generator = nn_util.DataGenerator(x_train, y_train, **self.config_training)

        self.loss_wout_reg = list(np.empty(epochs * training_generator.__len__()))
        self.loss = list(np.empty_like(self.loss_wout_reg))
        ind_loss = 0

        for epoch in range(epochs):

            print('Epoch: ', epoch)

            for X, y in training_generator:

                output = self.forward(X)
                loss = self.config_training['criterion'](output, y)
                self.loss_wout_reg[ind_loss] = float(loss)

                # add regularization terms to loss
                reg = torch.tensor(0., requires_grad=True)

                for param in self.parameters():
                    reg = reg + torch.linalg.vector_norm(param.flatten(), ord=self.config_training['regularization_ord'])**2
                
                loss = loss + self.config_training['regularization_alpha'] * reg
                self.loss[ind_loss] = float(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ind_loss += 1



    def init_config_training(self, **kwargs):

        default_extraction_strings = {
            'criterion': torch.nn.MSELoss(),
            'shuffle': True,
            'epochs': 1024, 
            'batch_size': 64,
            'regularization_alpha': 0.1, 
            'regularization_ord': 2,
            'learning_rate': 0.0001,
            'update_rule': torch.optim.Adam, 
        }

        config_training = nn_util.create_config(kwargs, default_extraction_strings)

        return config_training



    def plot1d(self, x, y, plot_xdim, plot_ydim):
        # use functions from plot functions script
        plt.plot(x[:,plot_xdim], y[:,plot_ydim], 'g.')
        plt.plot(x[:,plot_xdim], self.forward(x).detach()[:,plot_ydim], 'r.')
        plt.show()


    
    def plot2d(self, x, y, x0_min , x0_max, x1_min, x1_max, resolution=540, markersize=32, linewidth=0.5, max_plots=8, save=True, dirname=Path().cwd()):
        # d_in = 2, d_out = 1
        d_in = self.config_architecture['d_in']
        d_out = self.config_architecture['d_out']
        
        if d_in != 2:
            return ValueError('Input dimension of the model has to be 2 to apply the plot2d method.')

        x0, x1 = np.meshgrid(
            np.linspace(x0_min, x0_max, resolution),
            np.linspace(x1_min, x1_max, resolution)
        )
        grid = np.stack((x0, x1)).T.reshape(-1, 2)
        grid = torch.Tensor(grid).double()

        file_path = Path(dirname) / 'log1.txt'
        with open(file_path, 'w') as file:
            file.write('before forward pass')

        outs = self.forward(grid)

        file_path = Path(dirname) / 'log1.txt'
        with open(file_path, 'w') as file:
            file.write('after forward pass')

        if save:
            os.mkdir(Path(dirname))

        for i in range(min(max_plots, d_out)):

            fig = Figure()
            ax = fig.subplots()

            y_pred = outs.detach().numpy().T[i,:].reshape(resolution,resolution).T

            tmpcolornorm = colors.Normalize(
                vmin=float(min(np.min(y_pred), min(y[:,i]))),
                vmax=float(max(np.max(y_pred), max(y[:,i])))
            )
            im = ax.contourf(
                x0, 
                x1, 
                y_pred, 
                locator=ticker.LinearLocator(numticks=20), 
                norm=tmpcolornorm
            )
            ax.scatter(
                x[:,0], 
                x[:,1], 
                s=markersize, 
                c=y[:,i], 
                norm=tmpcolornorm, 
                edgecolors='w', 
                linewidth=linewidth
            )#,'ro')
            # myax.scatter(x_train[:,0],x_train[:,1],facecolor='none',edgecolor='black')
            fig.colorbar(im, ax=ax)

            if save:
                filename = Path(dirname) / 'fig_{}_2dplot.png'.format(i)
                fig.savefig(filename)
                fig.clf()

    

    def save(self, path):
        torch.save(self.state_dict(), path)
    


    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()