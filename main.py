from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--pooling', default='RPGaussian')
parser.add_argument('--dropout', default='0.7')
parser.add_argument('--epoch', default='70')


args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
# bz=1 # original
bz = 1
# lr=0.0005 # original
lr = 0.0005

pooling_type=args.pooling
dropout = float(args.dropout)
num_epochs = int(args.epoch)


# #gtea, random rp
#num_epochs = 50
#50 salads, random rp
#num_epochs = 70


print('-------------config:dataset={}, lr={}, ep_max={}, pooling={}, dropout={}----------'.format(args.dataset,lr, num_epochs,
	pooling_type, dropout))


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

# model_dir = "./models/"+args.dataset+"_{}_dropout{}_ep{}/split_".format(pooling_type,dropout,num_epochs)+args.split
# model_dir="./models/model_backup/"+args.dataset+"_gaussian_dropout{}_ep{}_right/split_".format(dropout,num_epochs)+args.split
# model_dir = "/home/yzhang/workspaces/ms-tcn-bilinear/models/model_backup/"+args.dataset+"_{}_dropout{}_ep{}_right/split_".format(pooling_type,dropout,num_epochs)+args.split
model_dir = "./models/"+args.dataset+"_{}_dropout{}_ep{}/split_".format(pooling_type,dropout,num_epochs)+args.split
results_dir = "./results/"+args.dataset+"_{}_dropout{}_ep{}/split_".format(pooling_type,dropout,num_epochs)+args.split


#results_dir = "./results/"+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes,
	pooling_type=pooling_type, dropout=dropout)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, 
    	device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
