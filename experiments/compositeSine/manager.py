import pytorch_lightning as pl

from experiments.compositeSine.logging_callback import LoggingCallback 

from core_code.create_data import CreateData
from core_code.create_model import CreateModel
from core_code.util.lightning import DataModule
from core_code.lightning_model import LightningModel

from experiments.util import BasicManager
from core_code.util.default_config import _make_init_config



class Manager(BasicManager):

    def __init__(self, configs_data, configs_architecture, configs_traininig, config_trainer, configs_custom):

        # self.set_randomness(configs_custom['torch_seed'], configs_custom['numpy_seed'])

        # save configs
        self.configs_data = configs_data 
        self.configs_architecture = configs_architecture
        self.configs_traininig = configs_traininig
        self.configs_trainer = config_trainer
        self.configs_custom = self.init_configs_custom(**configs_custom) # guarantee that it has the key determined by 'default_extraction_strings'

        self.configs_data_list, \
        self.configs_architecture_list, \
        self.configs_training_list, \
        self.configs_trainer_list, \
        self.configs_custom_list \
            = self._create_grid()    



    @staticmethod
    def init_configs_custom(**kwargs):

        default_extraction_strings = {
            'logger': True,
            'callbacks': True,
            'seed': 77,
            'workers': True,
            'logging_epoch_interval': 50
        }

        return _make_init_config(default_extraction_strings, **kwargs)[0]



    def run(
        self, 
        experimentbatch_name=None, 
        ind_configs=None, 
    ):

        self.configs_data_list, \
        self.configs_architecture_list, \
        self.configs_training_list, \
        self.configs_trainer_list, \
        self.configs_custom_list \
            = self._create_grid()

        if isinstance(ind_configs, type(None)):
            ind_configs = [i for i in range(self.num_experiments)]
        elif isinstance(ind_configs, int): # or isinstance(ind_configs, float):
            ind_configs = [int(ind_configs)]

        print(ind_configs)
        
        for i in ind_configs:
            
            # manage the grid configurations
            config_data = self.configs_data_list[i]
            config_architecture = self.configs_architecture_list[i]
            config_training = self.configs_training_list[i]
            config_custom = self.configs_custom_list[i]
            config_trainer = self.configs_trainer_list[i]

            # initialize the core objects#
            pl.seed_everything(config_custom['seed'], config_custom['workers'])
            data = CreateData(**config_data)
            torch_model = CreateModel(**config_architecture)
            data_module = DataModule(data, **config_training)

            model = LightningModel(torch_model, **config_training)

            logging_callback = LoggingCallback(
                *data.create_data('train').values(), 
                data.config_data, 
                logging_epoch_interval=config_custom['logging_epoch_interval']
            )
            
            model.fit(
                data_module,
                project="visualization", 
                name=experimentbatch_name + f'_config{i}',
                callbacks=[
                    logging_callback,
                ],
                **config_trainer
            )