'''
Starting on March 30, 2015
This script is based on 
/mnt/data/datasets/bugs_annotated_2014/Good Images
The outputs are 
/mnt/data/datasets/bugs_annotated_2014/AllGood
/mnt/data/datasets/bugs_annotated_2014/new_separation

new_separation will be used in final experiments

Splitting the data into training, validation, test in new ways.

rules to obey
    ratio of images with or without moth roughly the same
    average number of moths per image roughly the same
    training, validation, test set cannot have images from the same sequence
'''

import os
import shutil
import numpy as np
from annotation import get_annotation, get_bbs


def print_nums(xx):
    print "number of images: {}".format(len(xx))
    print "total number of moths: {}".format(sum(xx))
    print "number of moths per image {:.1f}".format(1. * sum(xx) / len(xx))
    print "number of images with moth: {}".format(np.sum(np.array(xx) > 0))
    print "number of images with no moth: {}".format(np.sum(np.array(xx) == 0))


def copy_folder(src, dst, src_files=None, ext=''):
    if not os.path.isdir(dst):
        os.makedirs(dst)
    if src_files is None:
        src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name + ext)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dst)


def copy_ann_jpg_png(src, dst_root, dst_sub, ann_list):
    copy_folder(src, os.path.join(dst_root, dst_sub), ann_list)
    copy_folder(src, os.path.join(dst_root, dst_sub), ann_list, '.jpg')
    copy_folder(src, os.path.join(dst_root, dst_sub), ann_list, '.png')


if __name__ == '__main__':

    flag_copy = False

    # images that are within the same sequence as the old annotated data
    list_rm = ["1621_35758", "1621_36808", "1622_37050", "1622_38714",
               "1622_41474", "1296_28989", "1296_32543", "1296_34313",
               "1296_35826", "1296_38467", "1471_34336", "1471_35844",
               "1471_38484", "1473_34342", "1475_35353", "1476_37913",
               "1235_30673"]

    dir_train = "/mnt/data/datasets/bugs_annotated_2014/Good Images/Training_Combined"
    dir_test = "/mnt/data/datasets/bugs_annotated_2014/Good Images/Test_Combined"
    dir_new = "/mnt/data/datasets/bugs_annotated_2014/Good Images/New_Test"
    dir_all = "/mnt/data/datasets/bugs_annotated_2014/AllGood"
    dir_newsep = "/mnt/data/datasets/bugs_annotated_2014/new_separation"

    ext_img = '.jpg'
    ext_seg = '.png'

    list_pos = [filename.strip(ext_img) for filename in os.listdir(dir_train)
                if ext_img in filename]
    list_neg = [filename.strip(ext_img) for filename in os.listdir(dir_test)
                if ext_img in filename]
    list_new = [filename.strip(ext_img) for filename in os.listdir(dir_new)
                if ext_img in filename]

    # test if the new test does not contain any images in the original data
    print set(list_neg + list_pos) & set(list_new)

    # test if there's an image to remove that
    print set(list_rm) - set(list_new)
    list_rm = list(set(list_rm) - (set(list_rm) - set(list_new)))

    if flag_copy:
        # step 1, move all the images into the same folder
        copy_folder(dir_new, dir_all)
        copy_folder(dir_train, dir_all)
        copy_folder(dir_test, dir_all)

        # step 2, remove new_test images that cost sequence problem
        for item in list_rm:
            os.remove(os.path.join(dir_all, item))
            os.remove(os.path.join(dir_all, item + ext_img))
            os.remove(os.path.join(dir_all, item + ext_seg))

    # step 3, make statistics by looking at the annotation files
    list_all = [filename.strip(ext_img) for filename in os.listdir(dir_all)
                if ext_img in filename]

    list_len = []

    for item in list_all:
        list_len.append(
            len(get_bbs(get_annotation(os.path.join(dir_all, item)))))

    # report statistics
    num_withmoth = np.sum(np.array(list_len) > 0)
    num_nomoth = np.sum(np.array(list_len) == 0)

    print ''
    print 'Total:'
    print_nums(list_len)

    # distribute
    # step sort, and then have probability of going to train and test

    # list_all_sorted,
    sorted_tuples = sorted(zip(list_len, list_all), reverse=True)
    list_ann_sorted = [img for (num, img) in sorted_tuples]
    list_len_sorted = [num for (num, img) in sorted_tuples]
    dict_img_len = {img: num for (num, img) in sorted_tuples}

    num_withmoth_train = 83
    num_withmoth_valid = 20
    num_withmoth_test = 30

    # num_nomoth_* should change according to the changes of num_withmoth_*
    num_nomoth_train = 27
    num_nomoth_valid = 7
    num_nomoth_test = 10

    assert num_withmoth_train + num_withmoth_valid + \
        num_withmoth_test == num_withmoth
    assert num_nomoth_train + num_nomoth_valid + \
        num_nomoth_test == num_nomoth

    list_train = []
    list_valid = []
    list_test = []

    np.random.seed(10)

    # dividing the with moth images
    perm_indices = np.random.permutation(range(num_withmoth))
    indices_withmoth_train = perm_indices[0:num_withmoth_train]
    indices_withmoth_valid = perm_indices[
        num_withmoth_train:num_withmoth_train + num_withmoth_valid]
    indices_withmoth_test = perm_indices[
        num_withmoth_train + num_withmoth_valid:num_withmoth]

    # dividing the no moth images
    perm_indices = np.random.permutation(range(num_withmoth,
                                               num_withmoth + num_nomoth))
    indices_nomoth_train = perm_indices[0: num_nomoth_train]
    indices_nomoth_valid = perm_indices[num_nomoth_train:
                                        num_nomoth_train + num_nomoth_valid]
    indices_nomoth_test = perm_indices[num_nomoth_train + num_nomoth_valid:
                                       num_nomoth]

    indices_train = list(indices_withmoth_train) + list(indices_nomoth_train)
    indices_valid = list(indices_withmoth_valid) + list(indices_nomoth_valid)
    indices_test = list(indices_withmoth_test) + list(indices_nomoth_test)
    # make statistics of train, test and valid
    # !!!! reuse the previous code for making statistics

    print ''
    print 'Training:'
    list_ann_train = np.array(list_ann_sorted)[indices_train]
    list_len_train = np.array(list_len_sorted)[indices_train]
    print_nums(list_len_train)
    print ''

    print 'Validation:'
    list_ann_valid = np.array(list_ann_sorted)[indices_valid]
    list_len_valid = np.array(list_len_sorted)[indices_valid]
    print_nums(list_len_valid)
    print ''

    print 'Test:'
    list_ann_test = np.array(list_ann_sorted)[indices_test]
    list_len_test = np.array(list_len_sorted)[indices_test]
    print_nums(list_len_test)
    print ''

    # copy images into different folder
    list_ann_withmoth_train = np.array(list_ann_sorted)[indices_withmoth_train]
    list_ann_nomoth_train = np.array(list_ann_sorted)[indices_nomoth_train]
    list_ann_withmoth_valid = np.array(list_ann_sorted)[indices_withmoth_valid]
    list_ann_nomoth_valid = np.array(list_ann_sorted)[indices_nomoth_valid]
    list_ann_withmoth_test = np.array(list_ann_sorted)[indices_withmoth_test]
    list_ann_nomoth_test = np.array(list_ann_sorted)[indices_nomoth_test]

    assert len(set(list(list_ann_withmoth_train) + list(list_ann_nomoth_train) +
                   list(list_ann_withmoth_valid) + list(list_ann_nomoth_valid) +
                   list(list_ann_withmoth_test) + list(list_ann_nomoth_test))) == \
        len(list_ann_withmoth_train) + len(list_ann_nomoth_train) + \
        len(list_ann_withmoth_valid) + len(list_ann_nomoth_valid) + \
        len(list_ann_withmoth_test) + len(list_ann_nomoth_test)

    # structures to make
    # train/withmoth
    # train/nomoth
    # valid/withmoth
    # valid/nomoth
    # test/withmoth
    # test/nomoth

    if flag_copy:

        copy_ann_jpg_png(dir_all, dir_newsep, 'train/withmoth',
                         list_ann_withmoth_train)
        copy_ann_jpg_png(dir_all, dir_newsep, 'train/nomoth',
                         list_ann_nomoth_train)
        copy_ann_jpg_png(dir_all, dir_newsep, 'valid/withmoth',
                         list_ann_withmoth_valid)
        copy_ann_jpg_png(
            dir_all, dir_newsep, 'valid/nomoth', list_ann_nomoth_valid)
        copy_ann_jpg_png(
            dir_all, dir_newsep, 'test/withmoth', list_ann_withmoth_test)
        copy_ann_jpg_png(
            dir_all, dir_newsep, 'test/nomoth', list_ann_nomoth_test)

        copy_folder(os.path.join(dir_newsep, 'train/withmoth'),
                    os.path.join(dir_newsep, 'train/combined'))
        copy_folder(os.path.join(dir_newsep, 'train/nomoth'),
                    os.path.join(dir_newsep, 'train/combined'))
        copy_folder(os.path.join(dir_newsep, 'valid/withmoth'),
                    os.path.join(dir_newsep, 'valid/combined'))
        copy_folder(os.path.join(dir_newsep, 'valid/nomoth'),
                    os.path.join(dir_newsep, 'valid/combined'))
        copy_folder(os.path.join(dir_newsep, 'test/withmoth'),
                    os.path.join(dir_newsep, 'test/combined'))
        copy_folder(os.path.join(dir_newsep, 'test/nomoth'),
                    os.path.join(dir_newsep, 'test/combined'))

        # making a toy dataset
        copy_ann_jpg_png(dir_all, dir_newsep, 'train_toy/withmoth',
                         list_ann_withmoth_train[:5])
        copy_ann_jpg_png(dir_all, dir_newsep, 'train_toy/nomoth',
                         list_ann_nomoth_train[:5])
        copy_ann_jpg_png(dir_all, dir_newsep, 'valid_toy/withmoth',
                         list_ann_withmoth_valid[:5])
        copy_ann_jpg_png(dir_all, dir_newsep, 'valid_toy/nomoth',
                         list_ann_nomoth_valid[:5])
        copy_ann_jpg_png(dir_all, dir_newsep, 'test_toy/withmoth',
                         list_ann_withmoth_test[:5])
        copy_ann_jpg_png(dir_all, dir_newsep, 'test_toy/nomoth',
                         list_ann_nomoth_test[:5])

        copy_folder(os.path.join(dir_newsep, 'train_toy/withmoth'),
                    os.path.join(dir_newsep, 'train_toy/combined'))
        copy_folder(os.path.join(dir_newsep, 'train_toy/nomoth'),
                    os.path.join(dir_newsep, 'train_toy/combined'))
        copy_folder(os.path.join(dir_newsep, 'valid_toy/withmoth'),
                    os.path.join(dir_newsep, 'valid_toy/combined'))
        copy_folder(os.path.join(dir_newsep, 'valid_toy/nomoth'),
                    os.path.join(dir_newsep, 'valid_toy/combined'))
        copy_folder(os.path.join(dir_newsep, 'test_toy/withmoth'),
                    os.path.join(dir_newsep, 'test_toy/combined'))
        copy_folder(os.path.join(dir_newsep, 'test_toy/nomoth'),
                    os.path.join(dir_newsep, 'test_toy/combined'))
