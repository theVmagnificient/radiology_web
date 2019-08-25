#!/bin/bash

docker ps | grep kafka | awk {'print $1'} | xargs docker stop
