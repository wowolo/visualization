import torch
import pytorch_lightning as pl

import core_code.util as util
from core_code.util.default_config import init_config_training
from core_code.util.config_extractions import _criterion_fm, _update_rule_fm

from core_code.lightning_util.data_module import DataModule


class LightningModel(pl.LightningModule):


    def __init__(self, model, **config_training):
        super(LightningModel, self).__init__()
        self.model = model # create model with model_selector
        self.config_training = init_config_training(**config_training)
        self.all_losses = util.check_config(**config_training)



    def forward(self, x):
        return self.model(x)



    def configure_optimizers(self):
        update = _update_rule_fm(self.config_training['update_rule'])
        optimizer = update(self.parameters(), lr=self.config_training['learning_rate'])
        return optimizer


    # TODO make convenience function used in the steps to go through each loss activity
    def training_step(self, batch, batch_idx):

        loss = torch.zeros((1), requires_grad=True)

        for loss_selec in batch.keys():

            loss_num = int(loss_selec.split('_')[1])
            loss_config = util.extract_lossconfig(self.config_training, loss_num)
            x, y, _loss_activity = batch['loss_{}'.format(loss_num)]

            output = self.forward(x)

            # compute loss based on loss configurations 
            _ind = (_loss_activity == loss_num)
            criterion = _criterion_fm(loss_config['criterion'])
            loss = loss + criterion(output[_ind], y[_ind])
        
        # add regularization terms to loss
        reg = torch.tensor(0., requires_grad=True)

        for param in self.parameters():
            reg = reg + torch.linalg.vector_norm(param.flatten(), ord=self.config_training['regularization_ord'])**2
        
        loss = loss + self.config_training['regularization_alpha'] * reg

        self.logger.experiment('train/loss', loss, on_epoch=True)

        return loss
    


    def validation_step(self, batch, batch_idx): # criterion without regularization on validation set
        
        loss = torch.zeros((1), requires_grad=True)

        for loss_selec in batch.keys():

            loss_num = int(loss_selec.split('_')[1])
            loss_config = util.extract_lossconfig(self.config_training, loss_num)
            x, y, _loss_activity = batch['loss_{}'.format(loss_num)]

            output = self.forward(x)

            # compute loss based on loss configurations 
            _ind = (_loss_activity == loss_num)
            criterion = _criterion_fm(loss_config['criterion'])
            loss = loss + criterion(output[_ind], y[_ind])
        
        self.logger.experiment('validation/loss', loss, on_step=False, on_epoch=True)

        return {'x': x, 'y': y, 'output': output}



    def test_step(self, batch, batch_idx): # criterion without regularization on test set
        
        loss = torch.zeros((1), requires_grad=True)

        for loss_selec in batch.keys():

            loss_num = int(loss_selec.split('_')[1])
            loss_config = util.extract_lossconfig(self.config_training, loss_num)
            x, y, _loss_activity = batch['loss_{}'.format(loss_num)]

            output = self.forward(x)

            # compute loss based on loss configurations 
            _ind = (_loss_activity == loss_num)
            criterion = _criterion_fm(loss_config['criterion'])
            loss = loss + criterion(output[_ind], y[_ind])

        self.logger.experiment('test/loss', loss, on_step=False, on_epoch=True)
        
        return loss

    

    def fit(self, data, logger=None, name=None, seed=None):
        # TODO add logging callbacks
        # TODO setup method such that it can be used with flags given via bash sript
    
        data_module = DataModule(data, **self.config_training)

        if isinstance(logger, type(None)):
            trainer = pl.Trainer(
                devices=6,
                # gpus=-1, 
                accelerator='cpu', 
                max_epochs=self.config_training['epochs'], 
                # deterministic=True,
                strategy="ddp_find_unused_parameters_false"
            )
        else:
            trainer = pl.Trainer(
                logger=logger,
                devices=6,
                # gpus=-1, 
                accelerator='cpu', 
                max_epochs=self.config_training['epochs'], 
                # deterministic=True,
                strategy="ddp_find_unused_parameters_false"
            )
        trainer.fit(self, data_module)