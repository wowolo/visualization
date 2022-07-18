import sys
# from experiments.compositeSine.management import ExperimentManager
from experiments.compositeSine.configs import configs_data, configs_architecture, configs_training, configs_custom
from experiments.compositeSine.logging_callback import LoggingCallback #, logging_callback

from core_code.create_data import CreateData
from core_code.create_model import CreateModel
from core_code.lightning_model import LightningModel

import wandb
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    try:
        experiment_name = sys.argv[1] 
    except IndexError:
        experiment_name = 'Stack_experiment_2'

    wandb.login()
    logger = WandbLogger(
        project = "visualization",
        name=experiment_name, 
        log_model=True
    )

    data = CreateData(**configs_data)
    torch_model = CreateModel(**configs_architecture)
    # datamodule = DataModule(data, **configs_training)
    model = LightningModel(torch_model, **configs_training)
    model.fit(data, logger=logger, callbacks=[LoggingCallback(*data.create_data('train').values(), data.config_data)])

    # data_module = DataModule(data, **configs_training)
    # trainer = Trainer(
    #     # logger=logger,
    #     devices=4,
    #     # gpus=-1, 
    #     accelerator="cpu", 
    #     max_epochs=configs_training['epochs'], 
    #     # deterministic=True,
    #     strategy='ddp', # "ddp_find_unused_parameters_false"
    # )
    # trainer.fit(model, data_module)



    # manager = ExperimentManager(
    #     configs_data, 
    #     configs_architecture, 
    #     configs_traininig,
    #     configs_custom # custom configs for chosen ExperimentManager
    # )
    # manager.run(experiment_name)