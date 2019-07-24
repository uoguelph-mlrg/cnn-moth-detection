from pipeline import proc_config, load_initial_data_wrap, load_misclf_data_wrap
from fileop import loadfile
import matplotlib.pylab as plt
import numpy as np
import random
from scipy.misc import imsave
import os
from config import FigExt


def concatenate_images(x_in, pad_size=1, num_row=10, num_col=10):
    x_pad = np.pad(x_in,
                   ((0, 0), (pad_size, pad_size),
                    (pad_size, pad_size), (0, 0)),
                   mode='constant', constant_values=0)

    x_cat = np.concatenate(
        [np.concatenate(x_pad[ind * num_row:(ind + 1) * num_row], axis=0)
         for ind in range(num_col)],
        axis=1)

    return x_cat


path = '/mnt/data/wding/tmp/bugs/bug_run_2015-04-14_20-58-36_39_single/misdetects0.pkl'

_, x_new_neg = loadfile(path)


config = loadfile('config.yaml')
config['flag_debug'] = True
config = proc_config(config)
config['dist_trans_list'] = (-5, 0, 5)
config['write_path'] = \
    '/mnt/data/wding/tmp/bugs/bug_run_2015-04-14_20-58-36_39_single'
x, y, _, _ = load_initial_data_wrap(config)
# x_new_neg, _ = load_misclf_data_wrap(config, ind_round=1, data_set='train')
_, x_new_neg = loadfile(path)
x_new_neg = np.array(x_new_neg) / 255.

x_reshaped = np.rollaxis(x.reshape((-1, 3, 28, 28)), 1, 4) / 255.

x_pos = x_reshaped[y.astype(bool)]
num_unique = x_pos.shape[0] / 72

ind_unique = 1
indices = []
for ind_rot in range(8):
    for ind_trans in range(9):
        indices.append(ind_rot * num_unique * 9 + ind_unique * 9 + ind_trans)

x_toshow = concatenate_images(x_pos[indices], num_col=24, num_row=3)
plt.imshow(x_toshow)

fig_dir = os.path.expanduser('~/Dropbox/automoth_paper_figures')
imsave(os.path.join(fig_dir, 'patches_aug' + FigExt), x_toshow, format=None)
