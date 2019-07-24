'''
For generating partial training data for performance evaluation
'''
import os
import random
import shutil
random.seed(1)

path_root = '/mnt/data/datasets/bugs_annotated_2014/new_separation'
path_pos_all = os.path.join(path_root, "train/withmoth")
path_neg_all = os.path.join(path_root, "train/nomoth")


def get_file_list(path):
    return sorted([f[:-4] for f in os.listdir(path) if '.jpg' in f])

for percent in [20, 40, 60, 80]:
    path_pos = os.path.join(path_root, "train_{}/withmoth".format(percent))
    path_neg = os.path.join(path_root, "train_{}/nomoth".format(percent))
    path_comb = os.path.join(path_root, "train_{}/combined".format(percent))

    if os.path.exists(path_pos):
        shutil.rmtree(path_pos)
    if os.path.exists(path_neg):
        shutil.rmtree(path_neg)
    if os.path.exists(path_comb):
        shutil.rmtree(path_comb)

    if not os.path.exists(path_pos):
        os.makedirs(path_pos)
    if not os.path.exists(path_neg):
        os.makedirs(path_neg)
    if not os.path.exists(path_comb):
        os.makedirs(path_comb)

    pos_list = get_file_list(path_pos_all)
    neg_list = get_file_list(path_neg_all)

    print 'number of negative: {}'.format(len(neg_list))
    print 'number of positive: {}'.format(len(pos_list))

    num_pos = 0
    for f in pos_list:
        if random.random() < (percent / 100.):
            shutil.copyfile(os.path.join(path_pos_all, f),
                            os.path.join(path_pos, f))
            shutil.copyfile(os.path.join(path_pos_all, f + '.jpg'),
                            os.path.join(path_pos, f + '.jpg'))
            shutil.copyfile(os.path.join(path_pos_all, f + '.png'),
                            os.path.join(path_pos, f + '.png'))
            shutil.copyfile(os.path.join(path_pos_all, f),
                            os.path.join(path_comb, f))
            shutil.copyfile(os.path.join(path_pos_all, f + '.jpg'),
                            os.path.join(path_comb, f + '.jpg'))
            shutil.copyfile(os.path.join(path_pos_all, f + '.png'),
                            os.path.join(path_comb, f + '.png'))

            num_pos += 1

    num_neg = 0
    for f in neg_list:
        if random.random() < (percent / 100.):
            shutil.copyfile(os.path.join(path_neg_all, f),
                            os.path.join(path_neg, f))
            shutil.copyfile(os.path.join(path_neg_all, f + '.jpg'),
                            os.path.join(path_neg, f + '.jpg'))
            shutil.copyfile(os.path.join(path_neg_all, f + '.png'),
                            os.path.join(path_neg, f + '.png'))
            shutil.copyfile(os.path.join(path_neg_all, f),
                            os.path.join(path_comb, f))
            shutil.copyfile(os.path.join(path_neg_all, f + '.jpg'),
                            os.path.join(path_comb, f + '.jpg'))
            shutil.copyfile(os.path.join(path_neg_all, f + '.png'),
                            os.path.join(path_comb, f + '.png'))
            num_neg += 1

    print 'num{}% of negative: {}, {}'.format(percent, num_neg,
                                              100. * num_neg / len(neg_list))
    print 'num{}% of positive: {}, {}'.format(percent, num_pos,
                                              100. * num_pos / len(pos_list))
