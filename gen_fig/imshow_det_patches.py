import os
import numpy as np
from copy import deepcopy
from scipy.misc import imread
from scipy.misc import imsave
from evaluation import evaluate_load
from evaluation import match_bbs
from data_munging import crop_centered_box

# det_file = '/mnt/data/wding/tmp/bugs/single_35/detections_test1.pkl'
data_path = \
    '/mnt/data/datasets/bugs_annotated_2014/new_separation/test/combined'
# write_path = '/mnt/data/wding/tmp/bugs/single_35'
# prob_threshold = 0.998
# bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path, det_file, n_images=-1)
# draw_bbs_on_img(bbs_dt_dict, bbs_gt_dict, data_path, write_path,
#                 prob_threshold=prob_threshold,
#                 overlap_threshold=0.5,
#                 plots_to_disk=True,
#                 data_set_name='test', ind_round=1,
#                 flag_pdf=True,)

det_file = '/mnt/data/wding/tmp/bugs/single_21/detections_test1.pkl'
write_path = '/mnt/data/wding/tmp/bugs/single_21'
prob_threshold = 0.823
bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path, det_file, n_images=-1)
# draw_bbs_on_img(bbs_dt_dict, bbs_gt_dict, data_path, write_path,
#                 prob_threshold=prob_threshold,
#                 overlap_threshold=0.5,
#                 plots_to_disk=True,
#                 data_set_name='test', ind_round=1,
#                 flag_pdf=True,)

# extract all miss and false positives


def get_box_with_new_size(old_bb, width, height=None):
    # [x1, y1, x2, y2], x left to right, y top to bottom

    if height is None:
        height = width

    hh = height / 2.
    hw = width / 2.

    x_c = (old_bb[0] + old_bb[2]) / 2.
    y_c = (old_bb[1] + old_bb[3]) / 2.

    new_bb = []
    new_bb.append(int(max(x_c - hw, 0)))
    new_bb.append(int(max(y_c - hh, 0)))
    new_bb.append(int(min(x_c + hw, 639)))
    new_bb.append(int(min(y_c + hh, 479)))

    return new_bb


def add_21x21_centerbox_to_patch(patch):
    patch = deepcopy(patch)
    patch_center = deepcopy(patch[40: 60, 40: 60, :])
    patch[39: 61, 39: 61, :] = 1
    patch[40: 60, 40: 60, :] = patch_center
    return patch




num_fp = 0
num_fn = 0

patch_size = 100

for img_name in bbs_dt_dict:

    bbs_dt = bbs_dt_dict[img_name]
    bbs_gt = bbs_gt_dict[img_name]
    img = imread(os.path.join(data_path, img_name))

    if prob_threshold > 0.:
        bbs_dt = np.array(
            [item for item in bbs_dt if item[4] > prob_threshold])

    #########################
    # calculate fppi and mr #
    matches, unmatched_dt, unmatched_gt = match_bbs(
        bbs_dt, bbs_gt, overlap=0.5)

    num_fp += len(unmatched_dt)
    num_fn += len(unmatched_gt)

    for ii, jj in enumerate(unmatched_dt):
        bb = bbs_dt[jj]
        x1, y1, x2, y2 = get_box_with_new_size(bb, patch_size)

        imsave('fps/{}_{}.png'.format(img_name[:-4], ii),
               add_21x21_centerbox_to_patch(img[y1: y2, x1: x2, :]))

    for ii, jj in enumerate(unmatched_gt):
        bb = bbs_gt[jj]
        x1, y1, x2, y2 = get_box_with_new_size(bb, patch_size)
        imsave('fns/{}_{}.png'.format(img_name[:-4], ii),
               add_21x21_centerbox_to_patch(img[y1: y2, x1: x2, :]))

    for ii, jj in enumerate(sorted(matches.values())):
        bb = bbs_gt[jj]
        x1, y1, x2, y2 = get_box_with_new_size(bb, patch_size)
        imsave('tps/{}_{}.png'.format(img_name[:-4], ii),
               add_21x21_centerbox_to_patch(img[y1: y2, x1: x2, :]))
