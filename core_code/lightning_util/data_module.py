import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from core_code.util.default_config import init_config_training
import core_code.util.helpers as util


class CustomDataset(Dataset):

    def __init__(self, x, y, loss_activity):
        
        self.x = x
        self.y = y
        self.loss_activity = loss_activity



    def __len__(self):
        return self.x.shape[0]



    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.loss_activity[idx]




def DataLoaders(x, y, loss_activity, **kwargs): 
    # check for tuple input -> loss activity
    allowed_keys = list(set(['batch_size', 'shuffle']).intersection(kwargs.keys()))
    dataloader_dict = {key: kwargs[key] for key in allowed_keys}
    dataloader_dict['batch_size']  = util.dict_extract(dataloader_dict, 'batch_size', 64)
    dataloader_dict['shuffle']  = util.dict_extract(dataloader_dict, 'shuffle', True)

    bool_data_loss_batching = util.dict_extract(kwargs, 'data_loss_batching', True) # default value

    # structure the generator by shuffle and data_loss_batching
    if bool_data_loss_batching:
        # determine the ratios based on given loss_activity and “total” batch size
        unique_activities = np.unique(loss_activity)
        _total_activities = [(loss_activity == i).sum() for i in unique_activities]
        _ratios = {'loss_{}'.format(i): float(_total_activities[i] / sum(_total_activities)) for i in range(len(_total_activities))}
        _max_ratio = np.argmax(_ratios.values())
        _max_ratio_key = list(_ratios.keys())[_max_ratio]

        # guarantee that batch size is sufficiently large to sample according to non-zero ratios
        _min_batch_size = sum([ratio > 0 for ratio in _ratios.values()])
        if dataloader_dict['batch_size'] < _min_batch_size:
            raise ValueError("Since 'data_loss_batching' is True and the loss_activity indicates that {} losses are used we need a total 'batch_size' of at least {}.".format(_min_batch_size, _min_batch_size))
        
        _batch_sizes = {key: max(1, int(_ratios[key] * dataloader_dict['batch_size'])) for key in _ratios.keys()}
        _batch_sizes_val = list(_batch_sizes.values())
        _batch_sizes[_max_ratio_key] = dataloader_dict['batch_size'] - sum(_batch_sizes_val[:_max_ratio] + _batch_sizes_val[_max_ratio+1:])
        _ind_lossdatas = [(loss_activity == i) for i in unique_activities]
        _dataset_partitions = {'loss_{}'.format(i): CustomDataset(x[_ind_lossdatas[i]], y[_ind_lossdatas[i]], loss_activity[_ind_lossdatas[i]]) for i in unique_activities}
        if not(isinstance(dataloader_dict['shuffle'], dict)):
            _shuffle = {'loss_{}'.format(i): dataloader_dict['shuffle'] for i in unique_activities}
        data_loaders = {'loss_{}'.format(i): DataLoader(_dataset_partitions['loss_{}'.format(i)], batch_size=_batch_sizes['loss_{}'.format(i)], shuffle=_shuffle['loss_{}'.format(i)]) for i in unique_activities}
    
    else:
        dataset = CustomDataset(x, y, loss_activity)
        data_loaders =  {'loss_0': DataLoader(dataset, **dataloader_dict)}

    return CombinedLoader(data_loaders)


class DataModule(pl.LightningDataModule):

    def __init__(self, data, **config_training):
        super().__init__()
        self.data = data
        self.config_training = init_config_training(**config_training)
        self.all_losses = util.check_config(**config_training)



    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = self.data.create_data('train') 
            self.data_val = self.data.create_data('val') 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = self.data.create_data('test') 



    def train_dataloader(self): # create training data based on 'data_loss_batching'
        data_loaders = DataLoaders(*self.data_train.values(), **self.config_training)
        return data_loaders



    def val_dataloader(self): 
        loader_config = {
            'batch_size': self.config_training['batch_size'], 
            'data_loss_batching': self.config_training['data_loss_batching'],
            'shuffle': False
        }
        data_loaders = DataLoaders(*self.data_val.values(), **loader_config)
        return data_loaders



    def test_dataloader(self): 
        loader_config = {
            'batch_size': self.config_training['batch_size'], 
            'data_loss_batching': self.config_training['data_loss_batching'],
            'shuffle': False
        }
        data_loaders = DataLoaders(*self.data_test.values(), **loader_config)
        return data_loaders
