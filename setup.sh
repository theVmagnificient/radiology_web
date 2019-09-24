#!/bin/bash

docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=:/data/web_volumes/archives \
  aimed_archives

docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=:/data/web_volumes/results \
  aimed_results
