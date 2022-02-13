#!/bin/bash

cd cooknet

DISPLAY=:0 nohup python3 -u cooklive.py ../cook_$(date -Idate).mp4 &> ../log_$(date -Idate).txt &
