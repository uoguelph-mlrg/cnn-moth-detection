'''
Try to evaluate the spread sheet
'''


import xlrd
import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl


def get_perform_measures(bbs_dt_dict, bbs_gt_dict, overlap=0.5,
                         img_shape=(480, 640)):

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


def match_and_sort_boxes(bbs_dt_dict, bbs_gt_dict, overlap=0.5):
    '''
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

    return box_prob_list, box_lookup_list, box_flagtp_list, box_flagfp_list


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


def add_to_dict(bbs_dict, img_name, tl_x, tl_y, br_x, br_y, prob=None):
    bb = (tl_x, tl_y, br_x, br_y, prob)

    if prob:
        bb += (prob, )
    bbs_dict[img_name].append(bb)


if __name__ == '__main__':
    filename = os.path.expanduser(
        '~/Downloads/test-20150409-threshold-0.95.xls')

    worksheet = xlrd.open_workbook(filename).sheet_by_index(0)

    bbs_dt_dict = {}
    bbs_gt_dict = {}

    for ind in range(1, worksheet.nrows):
        img_name = str(int(worksheet.cell_value(ind, 1)))
        bbs_dt_dict[img_name] = []
        bbs_gt_dict[img_name] = []


    for ind in range(1, worksheet.nrows):
        img_name = str(int(worksheet.cell_value(ind, 1)))
        flag_machine = bool(worksheet.cell_value(ind, 4))
        tl_x = int(worksheet.cell_value(ind, 5))
        tl_y = int(worksheet.cell_value(ind, 6))
        br_x = int(worksheet.cell_value(ind, 7))
        br_y = int(worksheet.cell_value(ind, 8))

        if flag_machine:
            prob = float(worksheet.cell_value(ind, 2))
            add_to_dict(bbs_dt_dict, img_name, tl_x, tl_y, br_x, br_y, prob)

        else:
            add_to_dict(bbs_gt_dict, img_name, tl_x, tl_y, br_x, br_y)

        # print img_name, prob, flag_machine, tl_x, tl_y, br_x, br_y

    for img_name in bbs_gt_dict:
        bbs_gt_dict[img_name] = np.array(bbs_gt_dict[img_name])
    for img_name in bbs_dt_dict:
        bbs_dt_dict[img_name] = np.array(bbs_dt_dict[img_name])

    thresh, fppi, miss_rate, recall, precision = \
        get_perform_measures(bbs_dt_dict, bbs_gt_dict)

    plt.figure()
    ax = plt.gca()
    ax.set_xticks(np.arange(0, max(fppi), 0.1))
    ax.set_yticks(np.arange(0, max(miss_rate), 0.1))
    plt.grid()
    plt.plot(fppi, miss_rate)
    plt.ylabel('miss rate')
    plt.xlabel('false positive per image')
    plt.show()
