import colorsys
import os
import yaml
import logging
import cPickle as pickle

import numpy as np
from scipy.misc import imresize, imread
from colorcorrect.algorithm import grey_world

from tools import sliding_window, nms, rot_img_array

logger = logging.getLogger(__name__)
imargs = {'cmap': 'gray', 'interpolation': 'none'}


def matlab_bbs_from_py(y_top, x_top, target_height, target_width):
    """ Takes bounding box defined by upper left corner, height and width
    and returns matlab-style bounding box in form x1, y1, x2, y2 """
    return (x_top + 1, y_top + 1, x_top + target_width, y_top + target_width)


def detect(data_path, write_path, target_width, target_height,
           x_stride, y_stride,
           thresh=0.5, n_images=-1,
           flag_rgb=True,
           flag_usemask=False,
           thresh_mask=0.05,
           nms_thresh=0.1,
           flag_det_rot_aug=False):
    """
    thresh is the detector confidence
    with conf > thresh we consider to be a detection
    if n_images is > 0, only look at the first n_images images
    """
    prob_maps, n_r, n_c = get_prob_maps(
        data_path, write_path,
        target_width, target_height,
        x_stride, y_stride,
        n_images=n_images,
        flag_rgb=flag_rgb,
        flag_usemask=flag_usemask,
        thresh_mask=thresh_mask,
        flag_det_rot_aug=flag_det_rot_aug)

    detections = thresh_and_nms(
        prob_maps, thresh, n_r, n_c, x_stride, y_stride,
        target_width, target_height, nms_thresh=nms_thresh)

    return detections
# end def detect


def thresh_and_nms_multiscale(detect_width_list,
                              detect_height_list,
                              prob_maps_scale_list,
                              n_r_scale_list, n_c_scale_list,
                              x_stride, y_stride,
                              target_width, target_height,
                              thresh, nms_thresh=0.1):

    bbs_list = []
    # for each scale do prob map, thresh and nms separately
    for ind_scale in range(len(detect_width_list)):

        detect_width = detect_width_list[ind_scale]
        detect_height = detect_height_list[ind_scale]
        prob_maps = prob_maps_scale_list[ind_scale]
        n_r = n_r_scale_list[ind_scale]
        n_c = n_c_scale_list[ind_scale]

        # here the stride should be the scaled stride on
        # the original image
        # and don't need to be integer
        x_scaled_stride = \
            1. * detect_width / target_width * x_stride
        y_scaled_stride = \
            1. * detect_height / target_height * y_stride
        bbs_1scale = thresh_and_nms(prob_maps, thresh, n_r, n_c,
                                    x_stride=x_scaled_stride,
                                    y_stride=y_scaled_stride,
                                    target_width=detect_width,
                                    target_height=detect_height,
                                    nms_thresh=nms_thresh)
        # here the target_* is actual window size

        bbs_list.append(bbs_1scale)
    # end for

    # then combine do nms on detect boxes from each scaled
    bbs = {}
    for img_name in bbs_list[0]:

        # iterating over to get the number of boxes, this can be optimized
        num_box = 0
        for ind_scale in range(len(bbs_list)):
            num_box += bbs_list[ind_scale][img_name].shape[0]

        bbs_this_img = np.zeros(
            (num_box, bbs_list[ind_scale][img_name].shape[1]))

        ind_box = 0
        for ind_scale in range(len(bbs_list)):
            bbs_this_img[ind_box: ind_box + bbs_list[ind_scale]
                         [img_name].shape[0]] = bbs_list[ind_scale][img_name]
            ind_box = ind_box + bbs_list[ind_scale][img_name].shape[0]

        # have not done further nms yet

        bbs[img_name], _ = nms(bbs_this_img, nms_thresh)

    return bbs
# end def thresh_and_nms_multiscale


def detect_multiscale(data_path, write_path, target_width, target_height,
                      x_stride, y_stride,
                      detect_width_list, detect_height_list,
                      thresh=0.5, n_images=-1,
                      flag_rgb=True,
                      flag_usemask=False,
                      thresh_mask=0.05,
                      nms_thresh=0.1,
                      flag_det_rot_aug=False):

    prob_maps_scale_list = []
    n_r_scale_list = []
    n_c_scale_list = []

    for ind_scale in range(len(detect_width_list)):

        detect_width = detect_width_list[ind_scale]
        detect_height = detect_height_list[ind_scale]

        prob_maps, n_r, n_c = get_prob_maps(
            data_path, write_path, target_width, target_height,
            x_stride, y_stride,
            detect_width=detect_width,
            detect_height=detect_height,
            n_images=n_images,
            flag_rgb=flag_rgb,
            flag_usemask=flag_usemask,
            thresh_mask=thresh_mask,
            flag_det_rot_aug=flag_det_rot_aug)

        prob_maps_scale_list.append(prob_maps)
        n_r_scale_list.append(n_r)
        n_c_scale_list.append(n_c)

    bbs = thresh_and_nms_multiscale(detect_width_list, detect_height_list,
                                    prob_maps_scale_list,
                                    n_r_scale_list, n_c_scale_list,
                                    x_stride, y_stride,
                                    target_width, target_height,
                                    thresh,
                                    nms_thresh=nms_thresh)

    return bbs
# end def detect_multiscale


def get_sliding_patch_features(data_path, write_path,
                               target_width, target_height,
                               x_stride, y_stride,
                               detect_width=None, detect_height=None,
                               n_images=-1,
                               flag_rgb=True,
                               img_ext='.jpg',
                               rot_kind=0):

    # detect_width and detect_height are for the directly snipped patches from
    # images.
    # target_width and target_height are the sizes of classifier input

    # determines if need to rescale and report
    flag_rescaling_test = False
    # determine if needed to rescale patches
    if detect_height == target_height and detect_width == target_width:
        print "Snipping patch sizes are equal to classifier input size."
    elif detect_height is None or detect_width is None:
        print "Patch sizes are not assigned. Assign them as classifier input size."
        detect_height, detect_width = target_height, target_width
    else:
        flag_rescaling_test = True
        print "Need to rescale patches to feed them into classifier."

    if not flag_rescaling_test:
        print "No need to rescale at test time."

    img_train = [img_f for img_f in os.listdir(
        data_path) if img_f.find(img_ext) > 0]

    # if n_images is specified, then only look at the first n_images
    if n_images > 1:
        img_train = img_train[:min(len(img_train), n_images)]

    patch_feature_dict = {}
    for ind_img, img_name in enumerate(img_train):
        try:
            im = imread(os.path.join(data_path, img_name))
        except IOError:
            logger.warn(
                "There was a problem reading the image: %s." % img_name)
            continue

        if im.ndim == 3:
            im = grey_world(im)

        # logger.debug("processing %s" % img_name)

        # print "processing %s" % img_name

        new_im_width = np.int(im.shape[1] * (1. * target_width / detect_width))
        new_im_height = np.int(
            im.shape[0] * (1. * target_width / detect_width))

        if flag_rgb:
            assert im.shape[2] == 3

            im = imresize(im, (new_im_height, new_im_width))

            # below is a slow implementation can remove
            # windows_list = []
            # for ind_rgb in range(3):
            # shape (65, 88, 28, 28)
            #     windows = sliding_window(im[:, :, ind_rgb],
            #                              win_height=target_height,
            #                              win_width=target_width,
            #                              x_stride=x_stride,
            #                              y_stride=y_stride)
            #     windows_list.append(windows)
            # shape (3, 65, 88, 28, 28)
            # windows = np.asarray(windows_list)

            # shape (65, 88, 28, 28, 3)
            windows = sliding_window(im,
                                     win_height=target_height,
                                     win_width=target_width,
                                     x_stride=x_stride,
                                     y_stride=y_stride)

            # shape (65, 88, 3, 28, 28)
            windows = np.rollaxis(windows, 4, 2)

            windows = rot_img_array(windows, kind=rot_kind)

            # shape (65, 88, 2352)
            windows = windows.reshape((windows.shape[0],
                                       windows.shape[1],
                                       windows.shape[2] * windows.shape[3]
                                       * windows.shape[4]))

        else:
            # the rollaxis command rolls the last (-1) axis back until the start
            # do a colourspace conversion

            if im.ndim == 3:
                im_y, im_i, im_q = colorsys.rgb_to_yiq(
                    *np.rollaxis(im[..., :3], axis=-1))
            else:
                im_y = im

            im_y = imresize(im_y, (new_im_height, new_im_width))

            windows = sliding_window(im_y, win_height=target_height,
                                     win_width=target_width,
                                     x_stride=x_stride,
                                     y_stride=y_stride)
            # print windows.shape

            windows = rot_img_array(windows, kind=rot_kind)

            windows = windows.reshape(
                (windows.shape[0], windows.shape[1],
                    windows.shape[2] * windows.shape[3]))

        # shape (2352 or 784, 65, 88)
        windows = np.rollaxis(windows, 2)

        # n_d number of feature dimension,
        # n_r number sliding windows in row direction
        # n_c number sliding windows in column direction
        n_d, n_r, n_c = windows.shape

        # shape (2352 or 784, 65 * 88)
        windows = windows.reshape(
            (windows.shape[0], windows.shape[1] * windows.shape[2]))

        # shape (65 * 88, 2352 or 784)
        X = windows.T

        patch_feature_dict[img_name] = X

        # above is sliding window

    return patch_feature_dict, n_r, n_c


# end def get_sliding_patch_features


def get_onerot_prob_maps(data_path, write_path,
                         target_width, target_height,
                         x_stride, y_stride,
                         clf=None, scaler=None,
                         detect_width=None, detect_height=None,
                         n_images=-1,
                         flag_rgb=True,
                         flag_usemask=False,
                         thresh_mask=0.05,
                         rot_kind=0):
    '''
    get prob_maps under a single rotation
    '''

    if clf is None or scaler is None:
        try:
            with open(os.path.join(write_path, 'clf.pkl'), 'rb') as f:
                clf, scaler = pickle.load(f)
        except IOError:
            raise IOError('No pretrained classifier found.')

    patch_feature_dict, n_r, n_c = \
        get_sliding_patch_features(data_path=data_path,
                                   write_path=write_path,
                                   target_width=target_width,
                                   target_height=target_height,
                                   x_stride=x_stride,
                                   y_stride=y_stride,
                                   detect_width=detect_width,
                                   detect_height=detect_height,
                                   n_images=n_images, flag_rgb=flag_rgb,
                                   rot_kind=rot_kind)

    if flag_usemask:
        patch_mask_dict, _, _ = \
            get_sliding_patch_features(data_path=data_path,
                                       write_path=write_path,
                                       target_width=target_width,
                                       target_height=target_height,
                                       x_stride=x_stride,
                                       y_stride=y_stride,
                                       detect_width=detect_width,
                                       detect_height=detect_height,
                                       n_images=n_images, flag_rgb=False,
                                       img_ext='.png',
                                       rot_kind=rot_kind)

    # probability map prediction
    prob_maps = {}
    for img_name in patch_feature_dict:
        X = patch_feature_dict[img_name]

        if flag_usemask:
            # assume jpg
            mask_name = img_name[:-4] + '.png'
            X_mask = patch_mask_dict[mask_name]

            # the True ones will be tested using convnet
            # the rest will just be set to zero
            valid_vec = ((255. - np.mean(X_mask, axis=1)) / 255.) > thresh_mask

            y_prob = np.ones((X.shape[0], 2))
            y_prob[:, 1] = 0.

            X = X[valid_vec]
            if X.shape[0] > 0:
                # output of scipy.misc.imresize is uint8, so there will be
                # warning here
                X = scaler.transform(np.cast['float32'](X))
                y_prob[valid_vec] = clf.predict_proba(X)

        else:
            # output of scipy.misc.imresize is uint8, so there will be warning
            # here
            X = scaler.transform(np.cast['float32'](X))
            y_prob = clf.predict_proba(X)

        prob_maps[img_name] = y_prob

    return prob_maps, n_r, n_c


def get_prob_maps(data_path, write_path,
                  target_width, target_height,
                  x_stride, y_stride,
                  detect_width=None, detect_height=None,
                  n_images=-1,
                  flag_rgb=True,
                  flag_usemask=False,
                  thresh_mask=0.05,
                  flag_det_rot_aug=False):
    """
    thresh is the detector confidence
    with conf > thresh we consider to be a detection
    if n_images is > 0, only look at the first n_images images

    if flag_det_rot_aug is True, then the prob maps will consider
    8 rotations of the original image
    """

    try:
        with open(os.path.join(write_path, 'clf.pkl'), 'rb') as f:
            clf, scaler = pickle.load(f)
    except IOError:
        raise IOError('No pretrained classifier found.')

        # from classification import train_simple_classifier
        # clf, scaler = train_simple_classifier()
        # with open(os.path.join(write_path, 'clf.pkl'), 'rb') as f:
        #     pickle.dump([clf, scaler], f)

    if not flag_det_rot_aug:
        return get_onerot_prob_maps(
            data_path=data_path, write_path=write_path,
            target_width=target_height, target_height=target_height,
            x_stride=x_stride, y_stride=y_stride,
            clf=clf, scaler=scaler,
            detect_width=detect_width, detect_height=detect_height,
            n_images=n_images,
            flag_rgb=flag_rgb,
            flag_usemask=flag_usemask,
            thresh_mask=thresh_mask,
            rot_kind=0)

    # if flag_det_rot_aug
    list_prob_maps = []
    num_rot_kind = 8

    print "rotation augmentation during detection"

    for rot_kind in range(num_rot_kind):
        prob_maps, n_r, n_c = get_onerot_prob_maps(
            data_path=data_path, write_path=write_path,
            target_width=target_height, target_height=target_height,
            x_stride=x_stride, y_stride=y_stride,
            clf=clf, scaler=scaler,
            detect_width=detect_width, detect_height=detect_height,
            n_images=n_images,
            flag_rgb=flag_rgb,
            flag_usemask=flag_usemask,
            thresh_mask=thresh_mask,
            rot_kind=rot_kind)
        list_prob_maps.append(prob_maps)

        print "rotation type: {}".format(rot_kind)

    # merge

    avg_prob_maps = {}
    for img_name in list_prob_maps[0]:
        avg_prob_maps[img_name] = list_prob_maps[0][img_name]

        for rot_kind in range(1, num_rot_kind):
            avg_prob_maps[img_name] += list_prob_maps[rot_kind][img_name]

        avg_prob_maps[img_name] /= num_rot_kind

    return avg_prob_maps, n_r, n_c

# end def get_prob_maps


def thresh_and_nms(prob_maps, thresh, n_r, n_c, x_stride, y_stride,
                   target_width, target_height, nms_thresh=0.1):

    n_detected_post_nms = 0
    n_detected = 0
    detections = {}

    for filename, y_prob in prob_maps.iteritems():

        y_pred = np.cast['int32'](y_prob[:, 1] > thresh)

        if y_pred.sum() > 0:
            logger.debug('detector scores at detections: \n' +
                         str(y_prob[y_pred == 1]))

        n_detected += y_pred.sum()

        # y_pred = np.cast['int32'](y_prob[:,1] > 0.8)

        # indices to predicted patches
        detections_linidx = (y_pred == 1).nonzero()[0]
        y_i, x_i = (y_pred == 1).reshape(n_r, n_c).nonzero()

        # get top left pixel for each patch
        y_top = y_i * y_stride
        x_top = x_i * x_stride

        # non-maximal suppression
        x2 = x_top + target_width
        y2 = y_top + target_height
        s = y_prob[detections_linidx][:, 1]  # 1 indicates the prob of moth
        bbs = np.c_[x_top, y_top, x2, y2, s]

        bbs1, pick = nms(bbs, nms_thresh)
        n_detected_post_nms += len(bbs1)

        detections[filename] = bbs1

    logger.info(
        'total # of detected bb: %d (pre) %d (post-nms)'
        % (n_detected, n_detected_post_nms))

    # detections is a dictionary
    # keys are name of images, values are n * 5 numpy.ndarray

    return detections
# end def thresh_and_nms


if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.load(f)

    detections = detect(data_path=os.path.join(config['data_path'],
                                               config['detect_test_set']),
                        write_path=config['write_path'],
                        target_width=config['target_width'],
                        target_height=config['target_height'],
                        x_stride=config['target_stride_x'],
                        y_stride=config['target_stride_y'],
                        nms_thresh=config['nms_threshhold'],
                        flag_det_rot_aug=config['flag_det_rot_aug'])

    with open(os.path.join(config['write_path'], 'detections.pkl'), 'wb') as f:
        pickle.dump(detections, f)
