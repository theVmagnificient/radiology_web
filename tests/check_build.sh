#!/bin/bash

echo Docker postfix: $1
num=$(docker ps -a --filter name=radiology_web_$1 --filter status=running | wc -l)

if (( num < 6 ))
then
	echo Not all container started
	echo Ecpected 6, got $num
	exit 1
else
	echo OK
	exit 0
fi
