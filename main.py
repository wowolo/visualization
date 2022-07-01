from experiments.compositeSine.management import ExperimentManager
from experiments.compositeSine.configs import configs_data, configs_architecture, configs_traininig, configs_custom

manager = ExperimentManager(
    configs_data, 
    configs_architecture, 
    configs_traininig,
    configs_custom # custom configs for chosen ExperimentManager
)
manager.run('Stack_experiment')