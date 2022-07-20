#!/bin/bash
echo $project_path
python "${project_path}/main.py" --experimentbatch_name $tag --config_trainer ${config_trainer[@]} --num_config $LSB_JOBINDEX