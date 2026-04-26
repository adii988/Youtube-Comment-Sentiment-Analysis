#!/bin/bash
docker stop yt-container || true
docker rm yt-container || true

docker pull 985539756441.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-plugin:latest

docker run -d -p 5000:5000 --name yt-container \
985539756441.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-plugin:latest