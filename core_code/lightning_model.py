import numpy as np
import torch
import pytorch_lightning as pl
import wandb

import core_code.util.helpers as util
from core_code.util.default_config import init_config_training
from core_code.util.config_extractions import _criterion_fm, _update_rule_fm

from core_code.util.lightning import DataModule


class LightningModel(pl.LightningModule):


    def __init__(self, model, **config_training):
        super(LightningModel, self).__init__()
        self.model = model # create model with model_selector
        self.config_training, self.all_tasks = init_config_training(**config_training)



    def forward(self, x):
        return self.model(x)



    def configure_optimizers(self):
        update = _update_rule_fm(self.config_training['update_rule'])
        optimizer = update(self.parameters(), lr=self.config_training['learning_rate'])
        return optimizer



    def training_step(self, batch, batch_idx):
        outputs = self._compute_combined_taskloss(batch)
        
        # add regularization terms to loss
        loss = outputs['loss']
        reg = torch.tensor(0., requires_grad=True)

        for param in self.parameters():
            reg = reg + torch.linalg.vector_norm(param.flatten(), ord=self.config_training['regularization_ord'])**2
        
        loss = loss + self.config_training['regularization_alpha'] * reg
        outputs['loss'] =  loss

        return outputs
    


    def validation_step(self, batch, batch_idx): # criterion without regularization on validation set
        outputs = self._compute_combined_taskloss(batch, bool_training=False)

        return outputs



    def test_step(self, batch, batch_idx): # criterion without regularization on test set
        outputs = self._compute_combined_taskloss(batch, bool_training=False)

        return outputs

    

    def fit(self, data, **config_trainer): #logger=None, name=None, seed=None, callbacks=None):
        # TODO setup method such that it can be used with flags given via bash sript - flags determined by trainer

        data_module = DataModule(data, **self.config_training)

        # trainer = pl.Trainer(**config_trainer)
        
        # if isinstance(logger, type(None)):
        trainer = pl.Trainer(
            devices=3,
            logger=config_trainer['logger'],
            callbacks=config_trainer['callbacks'],
            accelerator='cpu', 
            max_epochs=self.config_training['epochs'], 
            # deterministic=True,
            strategy="ddp_find_unused_parameters_false"
        )
        # else:
        #     trainer = pl.Trainer(
        #         logger=logger,
        #         devices=6,
        #         # gpus=-1, 
        #         accelerator='cpu', 
        #         max_epochs=self.config_training['epochs'], 
        #         # deterministic=True,
        #         strategy="ddp_find_unused_parameters_false"
        #     )

        trainer.fit(self, data_module)

        trainer.logger.experiment.config.update(data.config_data)
        trainer.logger.experiment.config.update(self.model.config_architecture)
        trainer.logger.experiment.config.update(self.config_training)
        trainer.logger.experiment.config.update(config_trainer)

    

    def _compute_combined_taskloss(self, batch, bool_training=True): # convenience function used in the step methods
        
        x, y, task_activity = self._retrieve_batch_data(batch)
        preds = self.forward(x)
        
        if not(bool_training):
            return {'preds': preds}

        else:
            # compute loss based on task configurations 
            loss = torch.zeros((1), requires_grad=True)
            unique_activities = torch.unique(task_activity).int()

            for task_num in unique_activities:

                task_config = util.extract_taskconfig(self.config_training, task_num)

                _ind = (task_activity == task_num)
                criterion = _criterion_fm(task_config['criterion'])
                loss = loss + criterion(preds[_ind], y[_ind])

            return {'preds': preds, 'loss': loss}


    
    def _retrieve_batch_data(self, batch):
        x = []
        y = []
        task_activity = []

        batch_keys = list(batch.keys())

        for task_key in batch_keys:

            _x, _y, _task_activity = batch[task_key]

            # concatenate the data to use in outputs
            x.append(_x)
            y.append(_y)
            task_activity.append(_task_activity)

        x = torch.concat(x)
        y = torch.concat(y)
        task_activity = torch.concat(task_activity)
        
        return x, y, task_activity