#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters (should be broker_name and topic_name)"
    exit 2
fi

target=($(cat docker-compose.yml | grep -oP "$1:[0-9].*$"))


name=${target[0]}
port="$(cut -d':' -f2 <<<"$name")"

echo "checking service with name $1 | topic ["$2"] | port $port"
docker-compose -f docker-compose.yml exec "$1" kafka-console-consumer --bootstrap-server localhost:$port --topic $2 --from-beginning
