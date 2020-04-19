import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os



action_list_gtea = [
'take',
'open',
'pour',
'close',
'shake',
'scoop',
'stir',
'put',
'fold',
'spread',
'background'
]


action_list_50salads = [
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


action_list_breakfast = [
'SIL',
'pour_cereals',
'pour_milk',
'stir_cereals',
'take_bowl',
'pour_coffee',
'take_cup',
'spoon_sugar',
'stir_coffee',
'pour_sugar',
'pour_oil',
'crack_egg',
'add_saltnpepper',
'fry_egg',
'take_plate',
'put_egg2plate',
'take_eggs',
'butter_pan',
'take_knife',
'cut_orange',
'squeeze_orange',
'pour_juice',
'take_glass',
'take_squeezer',
'spoon_powder',
'stir_milk',
'spoon_flour',
'stir_dough',
'pour_dough2pan',
'fry_pancake',
'put_pancake2plate',
'pour_flour',
'cut_fruit',
'put_fruit2bowl',
'peel_fruit',
'stir_fruit',
'cut_bun',
'smear_butter',
'take_topping',
'put_toppingOnTop',
'put_bunTogether',
'take_butter',
'stir_egg',
'pour_egg2pan',
'stirfry_egg',
'add_teabag',
'pour_water',
'stir_tea'
]





import sys
results_file_path = sys.argv[1] # e.g. 50salads_FirstOrder_dropout0.5_ep50/split_1/rgb-01-1

exp = results_file_path.split('/')[0]
dataset = exp.split('_')[0]
split = results_file_path.split('/')[1]
filename = results_file_path.split('/')[2]

if dataset == 'gtea':
    action_list = action_list_gtea
elif dataset == '50salads':
    action_list = action_list_50salads
elif dataset == 'breakfast':
    action_list = action_list_breakfast

gt_folder = '/home/yzhang/workspaces/ms-tcn-bilinear/data/{:s}/groundTruth'.format(dataset)
res_1_path = '/home/yzhang/workspaces/ms-tcn-bilinear/results/'+results_file_path


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


gt_content = read_file(os.path.join(gt_folder, filename+'.txt')).split('\n')[0:-1]

recog_content_res_1 = read_file(res_1_path).split('\n')[1].split()

action_ids_gt = [action_list.index(x) for x in gt_content]
action_ids_res1 = [action_list.index(x) for x in recog_content_res_1]


plt.figure('gt')
plt.pcolor(np.array(action_ids_gt).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_{:s}_{:s}_{:s}.png'.format('gt', dataset, filename))

plt.figure('gt')
plt.pcolor(np.array(action_ids_res1).reshape([1,-1]))
plt.savefig('/home/yzhang/Dropbox/vis_{:s}_{:s}.png'.format(exp, filename))
# plt.show()

