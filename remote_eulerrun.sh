#!/bin/bash
#
# main bash script for submitting a python/EULER job:
#     - allocate the resources
#     - load the python environment
#     - submit the job
#
# to run the script:
#     - execute "chmod +x python_job.sh"
#     - execute "./python_job.sh
#     - the argument "tag" is passed to the MATLAB code
#
############################################################################

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

# make connection to euler
ssh euler
username=scheins
# go to main directory and update via github
main_dir=/cluster/home/$username/master_thesis/visualization
cd $main_dir
git stash
git pull origin master

echo "#######################################################"
echo "start"
echo "#######################################################"

# ressource allocation
max_time="03:10" # maximum time (hour:second") allocated for the job (max 120:00 / large value implies low priority)
n_core="1" # number of core (large value implies low priority)
memory="65536" # memory allocation (in MB) per core (large value implies low priority)
scratch="0" # disk space (in MB) for temporary data per core


# get the job name (${1} is the tag provided as argument)
tag="${1}"

# get the log filename
log="${tag}_out.txt"
err="${tag}_error.txt"

# load python environment specified by second input
module load gcc/8.2.0 python_gpu/3.9.9
source /cluster/home/scheins/master_thesis/visualization/visual_env/bin/activate

# submit the job
bsub -J $tag -o $log -e $err -n $n_core -W $max_time -N -R "rusage[mem=$memory,scratch=$scratch]" python /cluster/home/scheins/master_thesis/visualization/main.py

# display the current queue
bbjobs


echo "#######################################################"
echo "end"
echo "#######################################################"

\exit

exit 0