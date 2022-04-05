#!/bin/bash

cd cooknet

export dname="../on_$(date -Idate)"

echo $dname
mkdir "$dname"

if [ $? -eq 1 ]
then
export dname=${dname}_1 #../cook_$(date -Idate)_1 #$(dname)_1
#echo "here"
mkdir -p $dname
echo "Saving files to $dname";
fi


DISPLAY=:0 nohup python3 -u cooklive.py $dname/cook_$(date -Idate).mp4 &> $dname/log_$(date -Idate).txt &
