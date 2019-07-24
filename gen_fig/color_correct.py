'''
For illustrating grey_world
'''

import os
import numpy as np
from colorcorrect.algorithm import grey_world
from scipy.misc import imread, imsave


def concatenate_images(img_list):
    temp_list = []
    for ind, img in enumerate(img_list):
        ind_col = ind % multi_size[1]
        if ind_col == 0:
            temp_list.append([img])
            if ind > 0:
                temp_list[-2] = np.concatenate(temp_list[-2], axis=1)
        else:
            temp_list[-1].append(img)

    temp_list[-1] = np.concatenate(temp_list[-1], axis=1)
    return np.concatenate(temp_list, axis=0)


raw_dir = os.path.expanduser('~/Work/automoth/gen_fig/raw')
proc_dir = os.path.expanduser('~/Work/automoth/gen_fig/proc')
fig_dir = os.path.expanduser('~/Dropbox/automoth_paper/figs')


name_list = [name for name in os.listdir(raw_dir)]
path_list = [os.path.join(raw_dir, name) for name in name_list]
img_list = map(imread, path_list)

img_proc_list = []
# process images
for img, name in zip(img_list, name_list):
    img_proc_list.append(grey_world(img))
    imsave(os.path.join(proc_dir, name), img_proc_list[-1])

# concatenate and down sample for output
multi_size = (4, 4)
orig_size = img_list[0].shape[:2]

img_raw_cat = np.zeros(tuple(np.array(multi_size) *
                             np.array(orig_size)) + (3,))

subsample = 2  # 2 or 4

imsave(os.path.join(fig_dir, 'colorcorrect_raw.png'),
       concatenate_images(img_list)[::subsample, ::subsample, :])
imsave(os.path.join(fig_dir, 'colorcorrect_proc.png'),
       concatenate_images(img_proc_list)[::subsample, ::subsample, :])
