#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi
echo "${1}" >> ./bash_files/tempfile.txt
cat ./bash_files/euler_commands.sh | ssh euler /bin/bash -s $1