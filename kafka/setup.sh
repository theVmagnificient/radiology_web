#!/bin/bash

if docker network ls | grep kafka-network; then
    echo kafka-network exists
else
    echo kafka-network does not exist, creating new one...
    docker network create kafka-network 
fi  

if docker volume ls | grep aimed_archives; then 
   echo volumes exist 
else 
   echo creating new volumes...
   docker volume create --name aimed_archives
   docker volume create --name aimed_results
   echo volumes created
fi

