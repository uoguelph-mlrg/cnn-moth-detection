import os
import yaml
import logging
import colorsys
import cPickle as pickle

import scipy.misc
import numpy as np
from sklearn import metrics
from colorcorrect.algorithm import grey_world
import matplotlib.pyplot as plt

from annotation import get_annotation, get_bbs
from data_munging import crop_and_rescale, crop_centered_box
from data_munging import augment_bbs_by_trans
from tools import x1y1x2y2_to_x1y1wh_batch
from tools import x1y1wh_to_x1y1x2y2_batch
from tools_plot import imshow_wrap, annotate_bbs, make_legend_wrap

logger = logging.getLogger(__name__)


def match_bbs(bbs_dt, bbs_gt, overlap=0.5):
    """
    bbs_dt: detected bounding boxes
    bbs_gt: ground truth bounding boxes
    overlap is the threshold for 2 bounding boxes to be considered a match
    """

    if bbs_dt.shape[0] == 0:
        # no detected bbs
        return {}, [], range(len(bbs_gt))

    elif bbs_gt.shape[0] == 0:
        return {}, range(len(bbs_dt)), []

    x1_dt = bbs_dt[:, 0]
    y1_dt = bbs_dt[:, 1]
    x2_dt = bbs_dt[:, 2]
    y2_dt = bbs_dt[:, 3]
    s = bbs_dt[:, -1]

    x1_gt = bbs_gt[:, 0]
    y1_gt = bbs_gt[:, 1]
    x2_gt = bbs_gt[:, 2]
    y2_gt = bbs_gt[:, 3]

    area_dt = (x2_dt - x1_dt) * (y2_dt - y1_dt)
    area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    # TODO: these may already be guaranteed to be sorted
    # unmatched_dt holds indices to detected but unprocessed bounding boxes in
    # increasing order of confidence

    unmatched_dt = range(len(bbs_dt))
    unmatched_gt = range(len(bbs_gt))

    # go through dt bounding boxes in decreasing order
    I = np.argsort(s)

    matches = {}

    for i in I[::-1]:
        # finds the intersection of each ground truth bounding box
        # with the current detected bounding box
        xx1 = np.fmax(x1_dt[i], x1_gt[unmatched_gt])
        yy1 = np.fmax(y1_dt[i], y1_gt[unmatched_gt])
        xx2 = np.fmin(x2_dt[i], x2_gt[unmatched_gt])
        yy2 = np.fmin(y2_dt[i], y2_gt[unmatched_gt])

        w = np.fmax(0.0, xx2 - xx1)
        h = np.fmax(0.0, yy2 - yy1)

        isc = w * h
        # Here using sum, which is problematic!!! FIXME
        # iou = isc / (area_dt[i] + area_gt[unmatched_gt])
        # iou = isc / area_gt[unmatched_gt]
        iou = isc / np.fmin(area_dt[i], area_gt[unmatched_gt])
        # iou = isc / (area_dt[i] + area_gt[unmatched_gt] - isc)

        # matches are when iou > overlap
        # if a bb_dt matches multiple bb_gt then the one with highest overlap
        if np.any(iou > overlap):
            matched_gt = unmatched_gt[np.argmax(iou)]
            # print "removing gt bb %d" % matched_gt
            unmatched_gt.remove(matched_gt)
            # print "removing dt bb %d" % i
            unmatched_dt.remove(i)

            matches[i] = matched_gt

    return matches, unmatched_dt, unmatched_gt
# end def match_bbs


def match_and_sort_boxes(bbs_dt_dict, bbs_gt_dict, overlap=0.5):
    '''
    Sort detections by probability values

    '''

    box_prob_list = []
    box_lookup_list = []
    box_flagtp_list = []
    box_flagfp_list = []

    for img_name in bbs_dt_dict:
        flag_tp = np.bool_(np.ones(len(bbs_dt_dict[img_name])))
        flag_fp = np.bool_(np.zeros(len(bbs_dt_dict[img_name])))
        _, unmatched, _ = match_bbs(bbs_dt_dict[img_name],
                                    bbs_gt_dict[img_name],
                                    overlap=overlap)

        flag_tp[unmatched] = False
        flag_fp[unmatched] = True

        box_flagtp_list += list(flag_tp)
        box_flagfp_list += list(flag_fp)

        for ind, box in enumerate(bbs_dt_dict[img_name]):
            box_prob_list.append(box[-1])
            box_lookup_list.append((img_name, ind))

    # sort these list according to prob
    box_prob_list, box_lookup_list, box_flagtp_list, box_flagfp_list = \
        [list(item) for item in zip(*sorted(
            zip(box_prob_list, box_lookup_list,
                box_flagtp_list, box_flagfp_list),
            reverse=True))]

    return (box_prob_list, box_lookup_list,
            box_flagtp_list, box_flagfp_list)


def evaluate(data_path, write_path, filename,
             num_neg_to_pick=0,
             prob_threshold=0., overlap_threshold=0.5,
             n_images=-1, flag_rescale=True, flag_rgb=True,
             target_width=32, target_height=32,
             flag_trans_aug=False,
             dist_trans_list=(-2, 0, 2),
             data_set_name='train', ind_round=0,
             ):

    # load the detected and ground truth bounding boxes
    bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path=data_path,
                                             filename=filename,
                                             n_images=n_images)

    return collect_false(
        bbs_dt_dict=bbs_dt_dict, bbs_gt_dict=bbs_gt_dict,
        data_path=data_path, write_path=write_path,
        num_neg_to_pick=num_neg_to_pick,
        prob_threshold=prob_threshold,
        overlap_threshold=overlap_threshold,
        data_set_name=data_set_name, ind_round=ind_round,
        flag_rescale=flag_rescale,
        target_width=target_width,
        target_height=target_height,
        flag_rgb=flag_rgb,
        flag_trans_aug=flag_trans_aug,
        dist_trans_list=dist_trans_list)
# end def evaluate


def get_fscore(recall, precision, beta=1.):
    return (1 + beta ** 2) * precision * \
        recall / (beta * precision + recall)


def get_perform_measures(bbs_dt_dict, bbs_gt_dict, overlap=0.5,
                         flag_gt_square=False,
                         gt_square_size=(28, 28),
                         img_shape=(480, 640)):

    # change the ground truth bounding boxes into squares for evaluation
    if flag_gt_square:
        target_height, target_width = gt_square_size
        img_height, img_width = img_shape
        bbs_gt_dict_orig = bbs_gt_dict
        bbs_gt_dict = {}
        for img_name in bbs_gt_dict_orig:
            bbs_orig = bbs_gt_dict_orig[img_name]
            bbs = []

            for bb_orig in bbs_orig:

                x_old_0, y_old_0, x_old_1, y_old_1 = bb_orig
                x_old_center = int((x_old_0 + x_old_1) / 2)
                y_old_center = int((y_old_0 + y_old_1) / 2)
                x_new_0 = min(img_width - target_width,
                              max(0,
                                  int(np.floor(x_old_center - target_width / 2.0))))
                y_new_0 = min(img_height - target_height,
                              max(0,
                                  int(np.floor(y_old_center - target_height / 2.0))))
                x_new_1 = x_new_0 + target_width
                y_new_1 = y_new_0 + target_height
                bbs.append((x_new_0, y_new_0, x_new_1, y_new_1))

            bbs_gt_dict[img_name] = np.array(bbs)

    box_prob_list, box_lookup_list, box_flagtp_list, box_flagfp_list = \
        match_and_sort_boxes(bbs_dt_dict, bbs_gt_dict, overlap)

    tp = np.cumsum(box_flagtp_list)
    fp = np.cumsum(box_flagfp_list)
    num_img = len(bbs_dt_dict)
    num_gt = sum([len(bbs_gt_dict[img_name]) for img_name in bbs_gt_dict])

    # beta > 1, recall more important, beta < 1 precision more important
    thresh = (
        np.array(box_prob_list[1:] + [0.]) + np.array(box_prob_list)) / 2.
    recall = 1. * tp / num_gt
    precision = 1. * tp / (tp + fp)
    miss_rate = 1 - recall
    fppi = 1. * fp / num_img

    return thresh, fppi, miss_rate, recall, precision


def get_perform_measures_from_file(data_path, filename, overlap=0.5,
                                   flag_gt_square=False, gt_square_size=(28, 28)):
    bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path, filename)
    return get_perform_measures(bbs_dt_dict, bbs_gt_dict, overlap,
                                flag_gt_square, gt_square_size)


def get_avg_miss_rate(fppi, miss_rate, auc_min=1, auc_max=10):

    valid_slice = np.logical_and(fppi > auc_min, fppi < auc_max)
    avg_miss_rate = metrics.auc(
        np.log(fppi[valid_slice]), miss_rate[valid_slice]) /\
        (np.log(auc_max) - np.log(auc_min))
    return avg_miss_rate


def load_bbs_from_ann(data_path, n_images=-1):
    '''
    load bounding boxes from annotation files
    '''

    jpg_train = [img_f for img_f in os.listdir(
        data_path) if img_f.find('.jpg') > 0]

    # if n_images is specified, then only look at the first n_images
    if n_images > 1:
        jpg_train = jpg_train[:min(len(jpg_train), n_images)]

    bbs_gt_dict = {}

    for ind_img, img_file in enumerate(jpg_train):

        logger.debug("processing %s" % img_file)

        ann_file = img_file.split('.')[0]
        ann_path = ann_path = os.path.join(data_path, ann_file)
        annotation = get_annotation(ann_path)

        # get all bbs for this image
        bbs_gt_raw = get_bbs(annotation)
        bbs_gt = np.empty((len(bbs_gt_raw), 4))
        for kk, (xy, width, height) in enumerate(bbs_gt_raw):
            x, y = xy
            bbs_gt[kk, :] = [x, y, x + width, y + height]

        bbs_gt_dict[img_file] = bbs_gt

    return bbs_gt_dict


def evaluate_load(data_path, filename, n_images=-1):
    """
    Pickle file at filename contains dictionary indexed by 
    image filename
    
    This contains a list of bounding boxes for each 
    images specifying detection
    """

    bbs_dt_dict = None
    # in case don't have detection file and only want to load ground truth
    # labels
    try:
        with open(filename, 'rb') as f:
            bbs_dt_dict = pickle.load(f)
    except IOError:
        logger.error("Couldn't load detections file")

    return bbs_dt_dict, load_bbs_from_ann(data_path, n_images=n_images)
# end def evaluate_load


def get_neg_to_pick_thresh(bbs, num):
    list_prob = sum([list(bbs[key][:, 4]) for key in bbs], [])
    list_prob = sorted(list_prob, reverse=True)
    if num >= len(list_prob):
        thresh = 0.
    else:
        thresh = (list_prob[num] + list_prob[num - 1]) / 2.
    return thresh


def collect_false(bbs_dt_dict, bbs_gt_dict,
                  data_path, write_path,
                  num_neg_to_pick=0,
                  prob_threshold=0.,
                  overlap_threshold=0.5,
                  data_set_name='train', ind_round=0,
                  flag_rescale=True,
                  target_width=32, target_height=32,
                  flag_rgb=True,
                  flag_trans_aug=False,
                  dist_trans_list=(-2, 0, 2),
                  ):
    '''
    Take detected bounding boxes and ground truth bounding boxes and output
    global statistics including:
        false positive per image
        missing rate
        precision at moth level
        recall at moth level

    '''

    neg_examples = []
    pos_examples = []

    # change prob_threshold, if num_neg_to_pick is specified
    if num_neg_to_pick > 0:
        prob_threshold = get_neg_to_pick_thresh(bbs_dt_dict, num_neg_to_pick)
        print "selected {} examples".format(num_neg_to_pick)
        print "thresh set to {}".format(prob_threshold)

    for img_name in bbs_dt_dict:

        bbs_dt = bbs_dt_dict[img_name]
        bbs_gt = bbs_gt_dict[img_name]

        if prob_threshold > 0.:
            bbs_dt = np.array(
                [item for item in bbs_dt if item[4] > prob_threshold])

        matches, unmatched_dt, unmatched_gt = match_bbs(
            bbs_dt, bbs_gt, overlap_threshold)

        try:
            im = scipy.misc.imread(os.path.join(data_path, img_name))
        except IOError:
            logger.warn(
                "There was a problem reading the jpg: %s." % img_name)
            continue

        im = grey_world(im)

        if flag_rgb:
            im_take = im
        else:
            im_y, im_i, im_q = colorsys.rgb_to_yiq(
                *np.rollaxis(im[..., :3], axis=-1))
            # the rollaxis command rolls the last (-1) axis back
            # until the start
            # do a colourspace conversion

            im_take = im_y

        def augment_bbs_by_trans_wrap(bbs_orig):
            bbs_totake = np.cast['int'](bbs_orig)[:, :4]
            if flag_trans_aug:
                bbs_totake = x1y1x2y2_to_x1y1wh_batch(bbs_totake)
                bbs_totake = augment_bbs_by_trans(
                    bbs_totake, dist_trans_list)
                bbs_totake = x1y1wh_to_x1y1x2y2_batch(bbs_totake)
            return bbs_totake

        def coord_valid(x1, y1, x2, y2, min_x=0, min_y=0,
                        max_x=im_take.shape[1], max_y=im_take.shape[0]):
            return x1 < min_x or y1 < min_y or x2 > max_x or y2 > max_y

        # gather negative examples from unmatched bb_dt
        if len(unmatched_dt) > 0:
            for x1, y1, x2, y2 in \
                    augment_bbs_by_trans_wrap(bbs_dt[unmatched_dt]):
                if coord_valid(x1, y1, x2, y2):
                    continue

                if y2 - y1 != target_height or x2 - x1 != target_width:
                    neg_examples.append(
                        scipy.misc.imresize(im_take[y1:y2, x1:x2], (target_height, target_width)))
                else:
                    neg_examples.append(im_take[y1:y2, x1:x2])

        # gather positive examples from unmatched bb_gt
        if len(unmatched_gt) > 0:
            for x1, y1, x2, y2 in \
                    augment_bbs_by_trans_wrap(bbs_gt[unmatched_gt]):
                if coord_valid(x1, y1, x2, y2):
                    continue

                # !!! here the target sizes should take parameters instead
                # of fixed 32 size.
                # determine if the xy, width, height are postive and within
                # range
                values_with_in_range = x2 - x1 > 0 and y2 - y1 > 0 \
                    and y1 >= 0 and y2 < im_take.shape[0] \
                    and x1 >= 0 and x2 < im_take.shape[1]

                if not values_with_in_range:
                    print "Bad boundingbox, ignored"
                    print x1, y1, x2 - x1, y2 - y1
                    continue

                if flag_rescale:
                    im_p = crop_and_rescale(im_take, (x1, y1), x2 - x1, y2 - y1,
                                            target_width=target_width, target_height=target_height)
                else:
                    im_p = crop_centered_box(im_take, (x1, y1), x2 - x1, y2 - y1,
                                             target_width=target_width, target_height=target_height)
                pos_examples.append(im_p)

    return pos_examples, neg_examples
# end def collect_false


def draw_bbs_on_img(bbs_dt_dict, bbs_gt_dict, data_path, write_path,
                    prob_threshold=0.,
                    overlap_threshold=0.5,
                    plots_to_disk=True,
                    data_set_name='train', ind_round=0,
                    flag_pdf=False,
                    ):
    '''
    Take detected bounding boxes and ground truth bounding boxes and output
    global statistics including:
        false positive per image
        missing rate
        precision at moth level
        recall at moth level

    '''

    if plots_to_disk:
        plt.ioff()
        pltdir_eval = os.path.join(
            write_path, 'plots_round-{}_{}_{:.3f}'.format(
                ind_round, data_set_name, prob_threshold))
        logger.info('writing eval plots to %s' % pltdir_eval)
        os.makedirs(pltdir_eval)
    else:
        plt.ion()

    plt.close('all')
    fig, subs = plt.subplots(nrows=1, ncols=2)

    for img_name in bbs_dt_dict:

        bbs_dt = bbs_dt_dict[img_name]
        bbs_gt = bbs_gt_dict[img_name]

        if prob_threshold > 0.:
            bbs_dt = np.array(
                [item for item in bbs_dt if item[4] > prob_threshold])

        #########################
        # calculate fppi and mr #
        matches, unmatched_dt, unmatched_gt = match_bbs(
            bbs_dt, bbs_gt, overlap_threshold)

        im = scipy.misc.imread(os.path.join(data_path, img_name))

        ########################
        # make plots if needed #

        imshow_wrap(subs[0], im)
        imshow_wrap(subs[1], im)
        annotate_bbs(subs[0], bbs_gt, 'green', None,
                     'labeled moths')

        annotate_bbs(subs[1], bbs_gt[unmatched_gt], 'blue',
                     None, 'misdetections')

        if bbs_dt.shape[0] > 0:  # can't index empty array
            annotate_bbs(subs[0], bbs_dt[:, :-1], 'magenta',
                         None, 'detections')
            make_legend_wrap(subs[0])

            annotate_bbs(subs[1], bbs_dt[unmatched_dt][:, :-1], 'red',
                         None, 'false positives')

            matched_dt = matches.keys()

            annotate_bbs(subs[1], bbs_dt[matched_dt][:, :-1], 'yellow',
                         None, 'correct detections')

            make_legend_wrap(subs[1])

        plt.tight_layout()

        if plots_to_disk:
            if flag_pdf:
                plt.savefig(
                    os.path.join(pltdir_eval, img_name + '.pdf'),
                    bbox_inches='tight', format='pdf')
            else:
                plt.savefig(
                    os.path.join(pltdir_eval, img_name), bbox_inches='tight')
        else:
            plt.draw()


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f)

    fppi, mr, pos_examples, neg_examples = \
        evaluate(data_path=os.path.join(config['data_path'],
                                        config['detect_test_set']),
                 write_path=config['write_path'],
                 filename=os.path.join(config['write_path'], 'detections.pkl'),
                 overlap_threshold=config['overlap_threshold'],
                 flag_rescale=config['flag_rescale'],
                 target_width=config['target_width'],
                 target_height=config['target_height'])
