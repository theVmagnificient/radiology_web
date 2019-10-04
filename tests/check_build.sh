#!/bin/bash

num=$(docker ps -a --filter name=radiology_web_jenkins --filter status=running | wc -l)

if (( num < 6 ))
then
	echo Not all container started
	exit 0
else
	echo OK
	exit 1
fi
