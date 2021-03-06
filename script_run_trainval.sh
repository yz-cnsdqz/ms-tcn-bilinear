#!/bin/bash

dataset=$1
pooling=$2
dropout=$3
epoch=$4

splits=$(seq 1 4)

if [ "$dataset" == 50salads ]
then
        splits=$(seq 1 5)
fi

for i in $splits;
do
	python main.py --action=train --dataset=$dataset --split=$i --pooling=$pooling --dropout=$dropout --epoch=$epoch
	python main.py --action=predict --dataset=$dataset --split=$i --pooling=$pooling --dropout=$dropout --epoch=$epoch
	echo $dataset training/eval done: split-"$i"
done

echo ---------------final results -------------------
python eval.py --dataset=$dataset --pooling=$pooling --dropout=$dropout --epoch=$epoch
