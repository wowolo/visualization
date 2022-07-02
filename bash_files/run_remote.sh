#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

# update experiments files on git
git add ./experiments/*
git commit -m "update experiments for running '${1}'"
git push origin master

# makes ssh connection and execute euler commands
cat ./bash_files/euler_commands.sh | ssh euler /bin/bash -s $1