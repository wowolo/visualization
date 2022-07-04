import os
from pathlib import Path
import numpy as np
import torch
import gc
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib import ticker, colors
import matplotlib.pyplot as plt

from core_code.nn_model.model_catalogue import ModelCatalogue
import core_code.nn_model.util as nn_util
from core_code.create_data import DataGenerators



class ExtendedModel(ModelCatalogue):

    @staticmethod
    def _criterions_fm(value):
        if isinstance(value, str):
            return {
                'MSELoss': torch.nn.MSELoss(),
            }[value]
        elif isinstance(value, tuple):
            args = value[1:]
            string = value[0]
            return {
                'dimred_MSELoss': nn_util.dimred_MSELoss(*args),
            }[string]
        else:
            raise ReferenceError('The function keyword is not yet implemented.')
    


    @staticmethod
    def _update_rule_fm(value):
        return {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
        }[value]



    def __init__(self, **kwargs):

        super(ExtendedModel, self).__init__(**kwargs) # pass appropriate input to create architecture
        
        # attempt to set to gpu
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print(device)

        self.loss = []
        self.loss_wout_reg = []



    def train(self, x_train, y_train, loss_activity=None, x_val=None, y_val=None, figure_path=Path().cwd(), **kwargs):

        if isinstance(self.config_architecture['architecture_key'], type(None)):
            return None

        self.config_training = self.init_config_training(**kwargs)

        if isinstance(loss_activity, type(None)):
            self.loss_activity = torch.ones(y_train.shape[0])
        elif isinstance(loss_activity, type(torch.Tensor())):
            self.loss_activity = loss_activity.int()
        else:
            raise ValueError('You have not supplied the loss activity in the necessary torch.Tensor format.')
        self.loss_activity = self.loss_activity.detach()

        # check that the loss_activity is compatible with criterions config
        max_loss = len(self.config_training['criterions'])
        min_entry = self.loss_activity.min()
        max_entry = self.loss_activity.max()
        if min_entry < 0 or max_entry > max_loss:
            raise ValueError('The entries of the loss activity assign losses between {} and {} while only entries between 0 and {} are possible.'.format(min_entry, max_entry, max_loss))
        bool_val_data = not(isinstance(x_val, type(None))) and not(isinstance(y_val, type(None)))
        if bool_val_data:
            if x_val.shape != x_train.shape:
                raise ValueError('The shape of the x training data ({}) and the shape of the x validation data ({}) do not match.'.format(x_val.shape, x_train.shape))
            if y_val.shape != y_train.shape:
                raise ValueError('The shape of the y training data ({}) and the shape of the y validation data ({}) do not match.'.format(y_val.shape, y_train.shape))

        epochs = self.config_training['epochs']    
        
        # prepare torch objects needed in training loop
        update = self._update_rule_fm(self.config_training['update_rule'])
        optimizer = update(self.parameters(), lr=self.config_training['learning_rate'])
        training_generators = DataGenerators(x_train, y_train, self.loss_activity, **self.config_training)
        if bool_val_data:
            val_generator = DataGenerators(x_val, y_val, self.loss_activity, **self.config_training)
        min_iter = min([training_generators[i].__len__() for i in range(len(training_generators))])

        self.loss_wout_reg = list(np.empty(epochs * min_iter)) 
        self.loss = list(np.empty_like(self.loss_wout_reg))
        if bool_val_data:
            self.val_loss = list(np.empty_like(self.loss_wout_reg))
        else:
            self.val_loss = None
        ind_loss = 0

        for epoch in range(epochs):

            data_retrievers = [iter(data_generator) for data_generator in training_generators]
            if bool_val_data:
                val_data_retrievers = [iter(data_generator) for data_generator in val_generator]

            for iteration in range(min_iter):
                
                loss = torch.tensor(0., requires_grad=True)
                if bool_val_data:
                    val_loss = torch.tensor(0., requires_grad=False)

                for i in range(len(data_retrievers)):

                    X, Y, temp_loss_activity = next(data_retrievers[i])

                    output = self.forward(X)
                    # compute loss based on config_training['criterions'] and loss activity
                    for loss_selec in range(max(temp_loss_activity)):
                        _ind = (temp_loss_activity == (loss_selec + 1))
                        criterion = self._criterions_fm(self.config_training['criterions'][loss_selec])
                        loss = loss + criterion(output[_ind], Y[_ind])
                        if loss < 0:
                            print('oh no')

                    if bool_val_data:
                        X_val, Y_val, temp_val_loss_activity = next(val_data_retrievers[i])
                        val_output = self.forward(X_val)
                        # compute loss based on config_training['criterions'] and loss activity
                        for loss_selec in range(max(temp_val_loss_activity)):
                            _ind = (temp_val_loss_activity == (loss_selec + 1))
                            criterion = self._criterions_fm(self.config_training['criterions'][loss_selec])
                            val_loss = val_loss + criterion(val_output[_ind], Y_val[_ind])
                
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


                if bool_val_data:
                    val_loss = val_loss + self.config_training['regularization_alpha'] * reg
                    self.val_loss[ind_loss] = float(val_loss)

                ind_loss += 1
            
            # print('Loss in epoch {}: {}'.format(epoch, loss))

            # training plot
            if self.config_training['loss_plot']:
                max_ind = (epoch + 1) * min_iter
                plt.figure()
                if self.config_training['regularization_alpha'] != 0:
                    plt.plot(self.loss_wout_reg[:max_ind], label='Training loss w/o regularization')
                    plt.plot(self.loss[:max_ind], label='Training loss with regularization')
                    n_col = 2
                    if bool_val_data:
                        plt.plot(self.val_loss[:max_ind], label='Validation loss with regularization')
                        n_col += 1
                else:
                    plt.plot(self.loss[:max_ind], label='Training loss')
                    n_col = 1
                    if bool_val_data:
                        plt.plot(self.val_loss[:max_ind], label='Validation loss')
                        n_col += 1
                plt.title('Training - Loss Plot')
                plt.xlabel('Step')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), fancybox=True, ncol=n_col)
                plt.savefig(figure_path / 'loss_plot.jpg', bbox_inches="tight")
                plt.close('all')
            




    def init_config_training(self, **kwargs):

        default_extraction_strings = {
            'criterions': 'MSELoss',
            'shuffle': True,
            'epochs': 1024, 
            'batch_size': 64,
            'regularization_alpha': 0.1, 
            'regularization_ord': 2,
            'learning_rate': 0.0001,
            'update_rule': 'Adam', 
            'separate_loss_batching': True,
            'loss_plot': True,
        }

        config_training = nn_util.create_config(kwargs, default_extraction_strings)

        if not(isinstance(config_training['criterions'], list)):
            config_training['criterions'] = [config_training['criterions']]

        return config_training



    def plot1d(self, x, y, plot_xdim, plot_ydim):
        # use functions from plot functions script
        plt.plot(x[:,plot_xdim], y[:,plot_ydim], 'g.')
        plt.plot(x[:,plot_xdim], self.forward(x).detach()[:,plot_ydim], 'r.')
        plt.show()


    
    def plot2d(self, x, y, x0_min , x0_max, x1_min, x1_max, resolution=540, markersize=32, linewidth=0.5, max_plots=8, save=True, dirname=Path().cwd()):
        
        if save:
            os.mkdir(Path(dirname))

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
                plt.cla() 
                plt.clf()
                plt.close("all")
                gc.collect()
        

    

    def save(self, path):
        torch.save(self.state_dict(), path)
    


    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()