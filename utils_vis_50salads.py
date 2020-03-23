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
results_file_path = sys.argv[1] # e.g. split_1/rgb-01-1

split = results_file_path.split('/')[0]
filename = results_file_path.split('/')[1]


gt_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/data/50salads/groundTruth'
res_1_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_RPGaussian_dropout0.0_ep50'
res_2_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_RPGaussian_dropout0.25_ep50'
res_3_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_RPGaussian_dropout0.5_ep50'
res_4_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/results/50salads_RPGaussian_dropout0.75_ep50'


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


gt_content = read_file(os.path.join(gt_folder, filename+'.txt')).split('\n')[0:-1]

recog_content_res_1 = read_file(os.path.join(res_1_folder,
                                            results_file_path)).split('\n')[1].split()
recog_content_res_2 = read_file(os.path.join(res_2_folder,
                                            results_file_path)).split('\n')[1].split()
recog_content_res_3 = read_file(os.path.join(res_3_folder,
                                            results_file_path)).split('\n')[1].split()
recog_content_res_4 = read_file(os.path.join(res_4_folder,
                                            results_file_path)).split('\n')[1].split()


action_ids_gt = [action_list.index(x) for x in gt_content]
action_ids_res1 = [action_list.index(x) for x in recog_content_res_1]
action_ids_res2 = [action_list.index(x) for x in recog_content_res_2]
action_ids_res3 = [action_list.index(x) for x in recog_content_res_3]
action_ids_res4 = [action_list.index(x) for x in recog_content_res_4]


plt.figure('gt')
plt.pcolor(np.array(action_ids_gt).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_gt4.png')

plt.figure('dropout 0')
plt.pcolor(np.array(action_ids_res1).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_dp04.png')


plt.figure('dropout 0.25')
plt.pcolor(np.array(action_ids_res2).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_dp0.254.png')


plt.figure('dropout 0.5')
plt.pcolor(np.array(action_ids_res3).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_dp0.54.png')


plt.figure('dropout 0.75')
plt.pcolor(np.array(action_ids_res4).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_dp0.754.png')

# plt.show()

