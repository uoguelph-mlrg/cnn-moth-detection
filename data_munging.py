import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
# import matplotlib as mpl
import os
import colorsys
import cv2
import logging
import itertools

from colorcorrect.algorithm import grey_world

from annotation import get_annotation, get_bbs
from tools_plot import dispims
from contours import get_contours
# from pprint import pprint

from scipy.misc import imresize

logger = logging.getLogger(__name__)

plt.ion()

imargs = {'cmap': 'gray', 'interpolation': 'none'}


def crop_centered_box(im, xy, width, height, target_width=32, target_height=32):
    """
    This function takes the same input and output format as the previous written
    crop_and_rescale. Therefore there are some of the variable names which is 
    not the true description of it.

    Given image and bounding box specified by xy, width, height
    Find a another box with target_height and target_width, which have the same 
    center as the given bounding box. It also deals with the boundary situation.
    """
    yx = np.cast['int32'](xy[::-1])
    shape = (int(height), int(width))

    target_x = int(np.floor(yx[1] + shape[1] / 2.0 - target_width / 2.0))
    target_y = int(np.floor(yx[0] + shape[0] / 2.0 - target_height / 2.0))

    if target_x < 0:
        target_x = 0
    elif target_x + target_width >= im.shape[1]:
        target_x = im.shape[1] - target_width

    if target_y < 0:
        target_y = 0
    elif target_y + target_height >= im.shape[0]:
        target_y = im.shape[0] - target_height

    im_crop = np.take(np.take(im,
                              np.arange(target_y, target_y + target_height),
                              axis=0),
                      np.arange(target_x, target_x + target_width), axis=1)

    return im_crop
# end def crop_centered_box


def crop_and_rescale_nearest(im, xy, width, height,
                             target_width=32, target_height=32,
                             detect_width_list=[8, 16, 32, 64],
                             detect_height_list=[8, 16, 32, 64],
                             ):
    '''
    For multiple scale detection, the cropped box for each groundtruth is the 
    smallest box that can fully cover the groundtruth label.
    And then scaled to the fixed size: target_width * target_height
    '''

    detect_width_list = np.array(detect_width_list)
    detect_height_list = np.array(detect_height_list)

    def find_ind(value, value_list):

        if value < max(value_list):
            ind = np.where(value_list >= value)[0][0]
        else:
            ind = len(value_list) - 1
        return ind

    ind_width = find_ind(width, detect_width_list)
    ind_height = find_ind(height, detect_height_list)
    ind = max(ind_width, ind_height)

    im_crop = crop_centered_box(im, xy, width, height,
                                target_width=detect_width_list[ind],
                                target_height=detect_height_list[ind])

    im_resized = imresize(im_crop, (target_height, target_width))

    if False:
        print ind_width
        print ind_height
        print im_crop.shape
        print im_resized.shape

    return im_resized
# end def crop_and_rescale_nearest


def crop_and_rescale(im, xy, width, height, target_width=32, target_height=32):
    """ Given image and bounding box specified by xy, width, height
    Crop region specified by bounding box and scale to target width and height """

    yx = np.cast['int32'](xy[::-1])
    shape = (int(height), int(width))
    small_dim = np.argmin(shape)  # pad smaller dimension
    large_dim = 1 - small_dim

    # pad up small dim so that we have square image
    pad_size = shape[large_dim] - shape[small_dim]

    pad_before = pad_size / 2
    pad_after = (pad_size / 2) + (pad_size % 2)  # extra goes at end

    small_bounds = (yx[small_dim] - pad_before,
                    yx[small_dim] + shape[small_dim] + pad_after)

    # bounds checking: did padding mean we exceed image dimensions?
    # if so, make window tight up against boundary
    if small_bounds[0] < 0:
        small_bounds = (0, shape[large_dim])

    if small_bounds[1] > im.shape[small_dim]:
        small_bounds = (im.shape[small_dim] - shape[large_dim],
                        im.shape[small_dim])

    # the min here is a fix for at least one of the annotations
    # which exceeds the image bounds
    large_bounds = (yx[large_dim], min(yx[large_dim] + shape[large_dim],
                                       im.shape[large_dim]))

    im_crop = np.take(np.take(im, np.arange(*small_bounds), axis=small_dim),
                      np.arange(*large_bounds), axis=large_dim)

    im_resized = imresize(im_crop, (target_height, target_width))

    return im_resized
# end def crop_and_rescale


def augment_bbs_by_trans(bbs, dist_trans_list):
    '''
    translation augmentation on bounding boxes
    '''

    xy_trans_list = tuple(itertools.product(dist_trans_list,
                                            dist_trans_list))

    bbs_orig = bbs
    bbs = []

    for (x, y), width, height in bbs_orig:
        for (x_trans, y_trans) in xy_trans_list:
            bbs.append(((x + x_trans, y + y_trans), width, height))

    return bbs


def get_pos(data_path, target_height, target_width,
            flag_rescale=False, flag_multiscale=False, flag_rgb=True,
            detect_width_list=[8, 16, 32, 64],
            detect_height_list=[8, 16, 32, 64],
            flag_trans_aug=False,
            dist_trans_list=(-2, 0, 2),
            ):
    """ Get positive training examples
    examples are rescaled to target_height and target_width

    With the assumption that the annotation file have the same name with 
    the image but with no extension

    flag_trans_aug: if do translation augmentation
    """

    jpg_train = [f for f in os.listdir(data_path) if f.find('.jpg') > 0]

    # moths = []
    moth_resized_list = []

    for i, j in enumerate(jpg_train):
        try:
            im = scipy.misc.imread(os.path.join(data_path, j))
        except IOError:
            logger.warn("There was a problem reading the jpg: %s." % j)
            continue

        im = grey_world(im)

        if not flag_rgb:
            # im will be assigned to the new gray image

            # the rollaxis command rolls the last (-1) axis back until the start
            # do a colourspace conversion
            im, im_i, im_q = colorsys.rgb_to_yiq(
                *np.rollaxis(im[..., :3], axis=-1))

        ann_file = j.split('.')[0]
        ann_path = os.path.join(data_path, ann_file)
        annotation = get_annotation(ann_path)

        # get all bbs for this image
        bbs = get_bbs(annotation)

        if flag_trans_aug:
            bbs = augment_bbs_by_trans(bbs, dist_trans_list)

        for xy, width, height in bbs:

            x, y = xy

            # determine if the xy, width, height are postive and within range
            values_with_in_range = width > 0 and height > 0 \
                and y >= 0 and y + height < im.shape[0] \
                and x >= 0 and x + width < im.shape[1]

            if not values_with_in_range:
                print "Bad boundingbox, ignored"
                print xy, width, height
                continue

            # remember y is indexed first in image
            # moth = im[y:(y + height), x:(x + width)]
            # moths.append(moth)
            # print moth.shape

            if flag_multiscale:

                moth_resized = crop_and_rescale_nearest(im, xy, width, height,
                                                        target_width, target_height,
                                                        detect_width_list=detect_width_list,
                                                        detect_height_list=detect_height_list)
            elif flag_rescale:

                moth_resized = crop_and_rescale(im, xy, width, height,
                                                target_width, target_height)
            else:
                moth_resized = crop_centered_box(im, xy, width, height,
                                                 target_width, target_height)
            moth_resized_list.append(moth_resized)

    return moth_resized_list
# end def get_pos


def get_neg(data_path, target_height, target_width, flag_rescale=True,
            flag_rgb=True, num_appr=2500,
            flag_trans_aug=False,
            dist_trans_list=(-2, 0, 2),
            ):
    '''
    generate negative training examples, which does not contain moths
    '''

    jpg_train = [f for f in os.listdir(data_path) if f.find('.jpg') > 0]

    blobs = []
    blob_resized_list = []

    for i, j in enumerate(jpg_train):
        try:
            im = scipy.misc.imread(os.path.join(data_path, j))
        except IOError:
            logger.warn("There was a problem reading the jpg: %s." % j)
            continue

        im = grey_world(im)

        # negative patches are extracted on color input image
        contours, c_area = get_contours(im)
        c_idx = np.argsort(c_area)[::-1]  # largest first
        contours = contours[c_idx]
        c_area = c_area[c_idx]

        # fig1, subs1 = plt.subplots(nrows=1, ncols=1)
        # cv2.drawContours(im, contours[:2], -1, (0, 255, 0), -1)

        # for c in xrange(2):
        # boundingRect returns top left corner xy
        # width, and height
        #     bx,by,bw,bh = cv2.boundingRect(contours[c])
        # cv2.rectangle(im,(bx,by),(bx+bw,by+bh),(255,0,0),3) # draw rectangle in blue color)
        # subs1.imshow(im, **imargs)
        # subs1.set_title(j)

        # plt.tight_layout()

        if not flag_rgb:
            # im will be assigned to the new gray image

            # the rollaxis command rolls the last (-1) axis back until the start
            # do a colourspace conversion
            im, im_i, im_q = colorsys.rgb_to_yiq(
                *np.rollaxis(im[..., :3], axis=-1))

        # get certain amount of bbs from this image based on the approximate
        # wanted number
        num_per_image = num_appr / len(jpg_train)
        if flag_trans_aug:
            num_per_image = num_per_image / (len(dist_trans_list) ** 2)

        bbs = []
        for c in contours[:num_per_image]:
            # boundingRect returns top left corner xy width, and height
            bx, by, bw, bh = cv2.boundingRect(c)
            xy = (bx, by)
            bbs.append([xy, bx, by])

        if flag_trans_aug:
            bbs = augment_bbs_by_trans(bbs, dist_trans_list)

        for (bx, by), bw, bh in bbs:
            # remember y is indexed first in image
            blob = im[by:(by + bh), bx:(bx + bw)]
            blobs.append(blob)
            # print moth.shape

            if flag_rescale:
                blob_resized = crop_and_rescale(im, xy, bw, bh,
                                                target_width, target_height)
            else:
                blob_resized = crop_centered_box(im, xy, bw, bh,
                                                 target_width, target_height)

            blob_resized_list.append(blob_resized)
    return blob_resized_list
# end def get_neg


def dispims_new(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 = int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout

    def gen_one_channel(M, height, width, bordercolor, border, n0, n1):
        im = bordercolor * \
            np.ones(
                ((height + border) * n0 + border, (width + border) * n1 + border), dtype='<f8')
        for i in range(n0):
            for j in range(n1):
                if i * n1 + j < M.shape[1]:
                    im[i * (height + border) + border:(i + 1) * (height + border) + border,
                       j * (width + border) + border:(j + 1) * (width + border) + border] = \
                        np.vstack((np.hstack((np.reshape(M[:, i * n1 + j], (height, width)),
                                              bordercolor * np.ones((height, border), dtype=float))),
                                   bordercolor *
                                   np.ones(
                                       (border, width + border), dtype=float)
                                   )
                                  )
        return im

    if M.ndim < 3:
        # the case of gray image or empty
        im = gen_one_channel(M, height, width, bordercolor, border, n0, n1)
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest', **kwargs)
    elif M.ndim == 3:
        im_list = []
        for ind in range(M.shape[0]):
            im_list.append(gen_one_channel(M[ind], height, width,
                                           bordercolor, border, n0, n1))
        im = np.transpose(np.array(im_list), axes=(1, 2, 0))

        # FIXME, the color display is not correct
        # im[:, :, 0], im[:, :, 1], im[:, :, 2] = \
        #     im[:, :, 0], im[:, :, 2], im[:, :, 1]

        plt.imshow(im, interpolation='nearest', **kwargs)

    # plt.show()
# end def dispims


if __name__ == "__main__":

    def show_examples(data_path, func,
                      target_height=32, target_width=32, flag_rgb=True):

        data_array = func(data_path, target_height, target_width)

        n_moths = len(data_array)

        data_array = np.asarray(data_array)
        if data_array.ndim < 4:
            m = data_array.reshape((n_moths, target_height * target_width))
        elif data_array.ndim == 4:
            m = data_array.reshape((n_moths, target_height * target_width, 3))

        plt.figure()
        dispims_new(m.T, target_height, target_width, border=2,
                    vmin=data_array.min(), vmax=data_array.max())

        plt.title('{} examples ({})'.format(data_path.split('/')[-1],
                                            data_path.split('/')[-2]))

        return data_array

    train_path_pos = '/mnt/data/datasets/bugs_annotated_2014/new_separation/train/withmoth'
    train_path_neg = '/mnt/data/datasets/bugs_annotated_2014/new_separation/train/nomoth'
    test_path_pos = '/mnt/data/datasets/bugs_annotated_2014/new_separation/test/withmoth'


    # FIXME, the color is incorrect
    data_train_pos = show_examples(train_path_pos, get_pos)
    data_train_neg = show_examples(train_path_neg, get_neg)
    data_test_pos = show_examples(test_path_pos, get_pos)
