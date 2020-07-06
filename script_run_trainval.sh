#!/bin/bash

#source 
#module load cuda/9.0
#module load cudnn/7.5-cu9.0


dataset=$1
pooling=$2

splits=$(seq 1 4)

if [ "$dataset" == 50salads ]
then
        splits=$(seq 1 5)
fi

for i in $splits;
do
	python main.py --action=train --dataset=$dataset --split=$i --pooling=$pooling
	python main.py --action=predict --dataset=$dataset --split=$i --pooling=$pooling
	echo $dataset training/eval done: split-"$i"
done

# echo ---------------final results -------------------
# python eval.py --dataset=$dataset --pooling=$pooling --dropout=$dropout --epoch=$epoch
