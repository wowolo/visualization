import sys
from experiments.compositeSine.management import ExperimentManager
from experiments.compositeSine.configs import configs_data, configs_architecture, configs_traininig, configs_custom

try:
    experiment_name = sys.argv[1] 
except IndexError:
    experiment_name = 'Stack_experiment'

manager = ExperimentManager(
    configs_data, 
    configs_architecture, 
    configs_traininig,
    configs_custom # custom configs for chosen ExperimentManager
)
manager.run(experiment_name)