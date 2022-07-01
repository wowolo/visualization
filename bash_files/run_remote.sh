#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

cat ./bash_files/euler_commands.sh | ssh euler /bin/bash -s $1