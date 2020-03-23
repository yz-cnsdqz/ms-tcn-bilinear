import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


action_list = [
    'add_dressing',
    'add_oil',
    'add_pepper',
    'add_salt',
    'add_vinegar',
    'cut_cheese',
    'cut_cucumber',
    'cut_lettuce',
    'cut_tomato',
    'mix_dressing',
    'mix_ingredients',
    'peel_cucumber',
    'place_cheese_into_bowl',
    'place_cucumber_into_bowl',
    'place_lettuce_into_bowl',
    'place_tomato_into_bowl',
    'serve_salad_onto_plate',
    'action_start',
    'action_end'
]


import sys
results_file_path = sys.argv[1]

split = results_file_path.split('/')[0]
filename = results_file_path.split('/')[1]


gt_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/data/50salads/groundTruth'
mstcn_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_mstcn'
rpb_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_RPBinary'
rpg_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_RPGaussian'



def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content



gt_content = read_file(os.path.join(gt_folder, filename+'.txt')).split('\n')[0:-1]

recog_content_mstcn = read_file(os.path.join(mstcn_folder,
                                            results_file_path)).split('\n')[1].split()
recog_content_rpb = read_file(os.path.join(rpb_folder,
                                            results_file_path)).split('\n')[1].split()
recog_content_rpg = read_file(os.path.join(rpg_folder,
                                            results_file_path)).split('\n')[1].split()


action_ids_gt = [action_list.index(x) for x in gt_content]
action_ids_mstcn = [action_list.index(x) for x in recog_content_mstcn]
action_ids_rpb = [action_list.index(x) for x in recog_content_rpb]
action_ids_rpg = [action_list.index(x) for x in recog_content_rpg]

plt.figure('gt')
plt.pcolor(np.array(action_ids_gt).reshape([1,-1]))

plt.figure('mstcn')
plt.pcolor(np.array(action_ids_mstcn).reshape([1,-1]))

plt.figure('rpb')
plt.pcolor(np.array(action_ids_rpb).reshape([1,-1]))


plt.figure('rpg')
plt.pcolor(np.array(action_ids_rpg).reshape([1,-1]))



            # edgecolors='k', linewidth=0.5)
plt.show()


