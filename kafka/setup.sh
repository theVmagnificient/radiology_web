#!/bin/bash

if docker network ls | grep kafka-network; then
    echo kafka-network exists
else
    echo kafka-network does not exist, creating new one...
    docker network create kafka-network 
fi  


