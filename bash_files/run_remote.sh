#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi
main_dir=~/dev/master_thesis/visualization
cat ./bash_files/euler_commands.sh | ssh euler /bin/bash