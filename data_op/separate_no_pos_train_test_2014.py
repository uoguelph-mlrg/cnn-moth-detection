'''
This script separate images and label files according to whether there are moths
labeled in the label file.

And then separate no counts and positive counts to training and test sets
respectively

And then combine to have a combined set of training image and a combined
set of test image
'''

import os
import shutil
from annotation import get_annotation, get_bbs
import numpy as np

DEBUG = False
dir_root = '/mnt/data/datasets/bugs_annotated_2014/Good Images'
dir_all = '/mnt/data/datasets/bugs_annotated_2014/Good Images/All'

dir_no_cnt_all = '/mnt/data/datasets/bugs_annotated_2014/Good Images/No_Counts_All'
dir_pos_cnt_all = '/mnt/data/datasets/bugs_annotated_2014/Good Images/Positive_Counts_All'
if not os.path.isdir(dir_no_cnt_all):
    os.makedirs(dir_no_cnt_all)    
if not os.path.isdir(dir_pos_cnt_all):
    os.makedirs(dir_pos_cnt_all)
    
# load the label file and determine no/pos counts and seperate to different 
# folders.
for filename in os.listdir(dir_all):
    if not filename.endswith(".jpg"):
        bbs = get_bbs(get_annotation(os.path.join(dir_all, filename)))
        
        if DEBUG:
            print bbs
        
        if len(bbs):
            shutil.copyfile(os.path.join(dir_all, filename), 
                            os.path.join(dir_pos_cnt_all, filename)
                            )
            shutil.copyfile(os.path.join(dir_all, filename + ".jpg"), 
                            os.path.join(dir_pos_cnt_all, filename + ".jpg")
                            )
        else:
            shutil.copyfile(os.path.join(dir_all, filename), 
                            os.path.join(dir_no_cnt_all, filename)
                            )
            shutil.copyfile(os.path.join(dir_all, filename + ".jpg"), 
                            os.path.join(dir_no_cnt_all, filename + ".jpg")
                            )


###############################################################################
                            
# separate the files into training and test set
dir_train_no_cnt = os.path.join(dir_root, 'No Counts/Training_Set')
dir_test_no_cnt = os.path.join(dir_root, 'No Counts/Test_Set')
dir_train_pos_cnt = os.path.join(dir_root, 'Positive Counts/Training Set')
dir_test_pos_cnt = os.path.join(dir_root, 'Positive Counts/Test Set')
if not os.path.isdir(dir_train_no_cnt):
    os.makedirs(dir_train_no_cnt)    
if not os.path.isdir(dir_test_no_cnt):
    os.makedirs(dir_test_no_cnt)
if not os.path.isdir(dir_train_pos_cnt):
    os.makedirs(dir_train_pos_cnt)    
if not os.path.isdir(dir_test_pos_cnt):
    os.makedirs(dir_test_pos_cnt)


ratio_train_test = 3

num_no_cnt = len(os.listdir(dir_no_cnt_all)) / 2
num_pos_cnt = len(os.listdir(dir_pos_cnt_all)) / 2

num_train_no_cnt = np.floor(num_no_cnt / (ratio_train_test + 1.0) * ratio_train_test)
# num_test_no_cnt = num_no_cnt - num_train_no_cnt

num_train_pos_cnt = np.floor(num_pos_cnt / (ratio_train_test + 1.0) * ratio_train_test)
# num_test_pos_cnt = num_pos_cnt - num_train_pos_cnt

idx_no_cnt = np.arange(num_no_cnt)
idx_pos_cnt = np.arange(num_pos_cnt)

np.random.seed(1)
np.random.shuffle(idx_no_cnt)
np.random.shuffle(idx_pos_cnt)

filelist_no_cnt = sorted(os.listdir(dir_no_cnt_all))
filelist_pos_cnt = sorted(os.listdir(dir_pos_cnt_all))

for i in idx_no_cnt:
    if i < num_train_no_cnt:
        shutil.copyfile(os.path.join(dir_no_cnt_all, filelist_no_cnt[2 * i]), 
                        os.path.join(dir_train_no_cnt, filelist_no_cnt[2 * i])
                        )
        shutil.copyfile(os.path.join(dir_no_cnt_all, filelist_no_cnt[2 * i + 1]), 
                        os.path.join(dir_train_no_cnt, filelist_no_cnt[2 * i + 1])
                        )
    else:
        shutil.copyfile(os.path.join(dir_no_cnt_all, filelist_no_cnt[2 * i]), 
                        os.path.join(dir_test_no_cnt, filelist_no_cnt[2 * i])
                        )
        shutil.copyfile(os.path.join(dir_no_cnt_all, filelist_no_cnt[2 * i + 1]), 
                        os.path.join(dir_test_no_cnt, filelist_no_cnt[2 * i + 1])
                        )

for i in idx_pos_cnt:
    if i < num_train_pos_cnt:
        shutil.copyfile(os.path.join(dir_pos_cnt_all, filelist_pos_cnt[2 * i]), 
                        os.path.join(dir_train_pos_cnt, filelist_pos_cnt[2 * i])
                        )
        shutil.copyfile(os.path.join(dir_pos_cnt_all, filelist_pos_cnt[2 * i + 1]), 
                        os.path.join(dir_train_pos_cnt, filelist_pos_cnt[2 * i + 1])
                        )
    else:
        shutil.copyfile(os.path.join(dir_pos_cnt_all, filelist_pos_cnt[2 * i]), 
                        os.path.join(dir_test_pos_cnt, filelist_pos_cnt[2 * i])
                        )
        shutil.copyfile(os.path.join(dir_pos_cnt_all, filelist_pos_cnt[2 * i + 1]), 
                        os.path.join(dir_test_pos_cnt, filelist_pos_cnt[2 * i + 1])
                        )

###############################################################################

# combine no/pos counts into train combined and test combined
dir_train_all = os.path.join(dir_root, 'Training_Combined')
dir_test_all = os.path.join(dir_root, 'Test_Combined')
if not os.path.isdir(dir_train_all):
    os.makedirs(dir_train_all)    
if not os.path.isdir(dir_test_all):
    os.makedirs(dir_test_all)

# combine training examples
for filename in os.listdir(dir_train_no_cnt):
    shutil.copyfile(os.path.join(dir_train_no_cnt, filename), 
                    os.path.join(dir_train_all, filename)
                    )
                    
for filename in os.listdir(dir_train_pos_cnt):
    shutil.copyfile(os.path.join(dir_train_pos_cnt, filename), 
                    os.path.join(dir_train_all, filename)
                    )
                    
# combine test examples
for filename in os.listdir(dir_test_no_cnt):
    shutil.copyfile(os.path.join(dir_test_no_cnt, filename), 
                    os.path.join(dir_test_all, filename)
                    )
                    
for filename in os.listdir(dir_test_pos_cnt):
    shutil.copyfile(os.path.join(dir_test_pos_cnt, filename), 
                    os.path.join(dir_test_all, filename)
                    )
