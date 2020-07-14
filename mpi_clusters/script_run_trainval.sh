#!/bin/bash


#source /is/ps2/yzhang/virtualenv/python3-torch1.1.0-mstcn/bin/activate

#source /is/ps2/yzhang/virtualenv/python2-torch-0.4.1-mstcn/bin/activate
#module load cuda-9.0
#module load cudnn-7.5-cu9.0
source /is/ps2/yzhang/virtualenv/python3-torch-1.2/bin/activate
#module load cuda/10.0
#module load cudnn-7.5-cu10.0

dataset=$1
pooling=$2
dropout=$3
epoch=$4
split=$(($5+1))
seedid=$6



python main.py --action=train --dataset=$dataset --split=$split --pooling=$pooling --dropout=$dropout --epoch=$epoch --seedid=$seedid
python main.py --action=predict --dataset=$dataset --split=$split --pooling=$pooling --dropout=$dropout --epoch=$epoch --seedid=$seedid
        #python eval.py --dataset=$dataset --split=$i --pooling=$pooling
#echo $dataset training/eval done: split-"$i"
deactivate
