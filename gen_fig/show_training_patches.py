from pipeline import proc_config, load_initial_data_wrap, load_misclf_data_wrap
from fileop import loadfile
import matplotlib.pylab as plt
import numpy as np
import random
from scipy.misc import imsave
import os
from tools_plot import concatenate_images




path = '/mnt/data/wding/tmp/bugs/bug_run_2015-04-14_20-58-36_39_single/misdetects0.pkl'

_, x_new_neg = loadfile(path)


config = loadfile('config.yaml')
config['flag_debug'] = True
config['flag_rot_aug'] = False
config['flag_trans_aug'] = False
config = proc_config(config)
config['write_path'] = '/mnt/data/wding/tmp/bugs/bug_run_2015-04-14_20-58-36_39_single'
x, y, _, _ = load_initial_data_wrap(config)
# x_new_neg, _ = load_misclf_data_wrap(config, ind_round=1, data_set='train')
_, x_new_neg = loadfile(path)
x_new_neg = np.array(x_new_neg) / 255.

x_reshaped = np.rollaxis(x.reshape((-1, 3, 28, 28)), 1, 4) / 255.

num_img = 100

x_pos = x_reshaped[y.astype(bool)]
x_neg = x_reshaped[np.logical_not(y)]

random.seed(0)
indices_rand = random.sample(range(len(x_pos)), 100)
x_pos = x_pos[indices_rand]
indices_rand = random.sample(range(len(x_neg)), 100)
x_neg = x_neg[indices_rand]

pad_size = 1
num_row = 10
num_col = 10


x_pos_cat = concatenate_images(x_pos)
x_neg_cat = concatenate_images(x_neg)
x_new_neg_cat = concatenate_images(x_new_neg[:num_img])

fig_dir = os.path.expanduser('~/Dropbox/automoth_paper_figures')

imsave(os.path.join(fig_dir, 'patches_pos.png'), x_pos_cat, format=None)
imsave(os.path.join(fig_dir, 'patches_neg.png'), x_neg_cat, format=None)
imsave(os.path.join(fig_dir, 'patches_new_neg.png'), x_new_neg_cat, format=None)

# plt.figure()
# plt.imshow(x_pos_cat)
# plt.axis('off')
# plt.tight_layout()

# plt.figure()
# plt.imshow(x_neg_cat)
# plt.axis('off')
# plt.tight_layout()

# plt.figure()
# plt.imshow(x_new_neg_cat)
# plt.axis('off')
# plt.tight_layout()

# plt.show()
