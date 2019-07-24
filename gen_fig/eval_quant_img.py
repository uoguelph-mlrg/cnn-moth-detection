from evaluation import evaluate_load, get_fscore
import os
import numpy as np
import matplotlib.pyplot as plt
from fileop import loadfile
from pipeline import proc_config
from sklearn import metrics
import matplotlib

from config import FigExt


def get_image_level_performance_from_file(data_path, filename, overlap=0.5, beta=2.):
    bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path, filename)
    return get_image_level_performance(bbs_dt_dict, bbs_gt_dict, overlap)


def get_image_level_performance(bbs_dt_dict, bbs_gt_dict, overlap=0.5, beta=2.):
    assert set(bbs_dt_dict.keys()) == set(bbs_gt_dict.keys())

    thresholds = [item / 100. for item in range(101)]
    prob_array = np.array(
        sorted(sum([list(bbs[:, 4]) for bbs in bbs_dt_dict.values()], [])))
    thresholds = (prob_array[:-1] + prob_array[1:]) / 2.

    tps, fps, tns, fns = [], [], [], []

    for thresh in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for img_name in bbs_gt_dict:

            flag_gt = bool(len(bbs_gt_dict[img_name]))
            flag_dt = bool(np.sum(bbs_dt_dict[img_name][:, 4] > thresh))
            # print len(bbs_dt_dict[img_name])

            if not flag_gt and not flag_dt:
                tn += 1
            elif not flag_gt and flag_dt:
                fp += 1
            elif flag_gt and not flag_dt:
                fn += 1
            elif flag_gt and flag_dt:
                tp += 1

        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

    tp = np.array(tps)
    fp = np.array(fps)
    tn = np.array(tns)
    fn = np.array(fns)

    sensitivity = 1. * tp / (tp + fn)
    specificity = 1. * tn / (tn + fp)
    recall = sensitivity
    precision = 1. * tp / (tp + fp)

    indices_valid = np.logical_not(np.isnan(precision))
    precision = precision[indices_valid]
    recall = recall[indices_valid]
    thresholds_pr = thresholds[indices_valid]

    if min(recall) > 0.:
        recall = np.concatenate((recall, [0.]))
        precision = np.concatenate((precision, [1.]))
        thresholds_pr = np.concatenate((thresholds_pr, [1.]))

    fscore = get_fscore(recall, precision, beta=beta)

    mcc = (tp * tn - fp * fn) / \
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    precision_recall_auc = metrics.auc(
        recall, precision, reorder=True)

    sens_spec_auc = metrics.auc(
        specificity, sensitivity, reorder=True)

    return dict(
        sensitivity=sensitivity,
        specificity=specificity,
        recall=recall,
        precision=precision,
        precision_recall_auc=precision_recall_auc,
        sens_spec_auc=sens_spec_auc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        thresholds=thresholds,
        thresholds_pr=thresholds_pr,
        fscore=fscore,
        mcc=mcc,
    )


def get_image_level_performance_wrap(run_path, det_file, overlap=0.5, beta=2.):
    config = loadfile(os.path.join(run_path, 'config.yaml'))
    config = proc_config(config)
    detections_test_file = os.path.join(run_path, det_file)

    return get_image_level_performance_from_file(
        os.path.join(config['data_path'], config['detect_test_set']),
        detections_test_file,
        overlap=overlap,
        beta=beta)


if __name__ == '__main__':
    fig_dir = os.path.expanduser('~/Dropbox/automoth_paper_figures')

    list_run_path = [
        '/mnt/data/wding/tmp/bugs/logreg_21',
        '/mnt/data/wding/tmp/bugs/logreg_28',
        '/mnt/data/wding/tmp/bugs/logreg_35',
        '/mnt/data/wding/tmp/bugs/logreg_42',
        '/mnt/data/wding/tmp/bugs/logreg_49',
        '/mnt/data/wding/tmp/bugs/single_21',
        '/mnt/data/wding/tmp/bugs/single_28',
        '/mnt/data/wding/tmp/bugs/single_35',
        '/mnt/data/wding/tmp/bugs/single_42',
        '/mnt/data/wding/tmp/bugs/single_49',
        '/mnt/data/wding/tmp/bugs/21_no_aug',
        '/mnt/data/wding/tmp/bugs/21_only_rot',
        '/mnt/data/wding/tmp/bugs/21_only_trans',
    ]

    list_det_file = [
        'detections_test0.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
        'detections_test1.pkl',
    ]

    list_labels = [
        'LogReg 21',
        'LogReg 28',
        'LogReg 35',
        'LogReg 42',
        'LogReg 49',
        'ConvNet 21',
        'ConvNet 28',
        'ConvNet 35',
        'ConvNet 42',
        'ConvNet 49',
        '21_no_aug',
        '21_only_rot',
        '21_only_trans',
    ]

    label2style_dict = {
        'ConvNet 35': 'g-',
        'LogReg 42': 'b--',
    }

    list_results = []

    font = {'family': 'normal',
            'size': 20}

    matplotlib.rc('font', **font)

    for run_path, det_file in zip(list_run_path, list_det_file):
        list_results.append(
            get_image_level_performance_wrap(run_path, det_file, overlap=0.5))

    for results, label in zip(list_results, list_labels):
        print "{}: precision-recall AUC: {:.3f}".format(
            label, results['precision_recall_auc'])
        print "{}: sensitivity-specificity AUC: {:.3f}".format(
            label, results['sens_spec_auc'])

    # plt.figure()
    # plt.hold('on')

    # for key in ['tp', 'fn', 'fp', 'tn']:
    #     plt.plot(results['thresholds'], results[key], label=key)
    #     plt.legend(loc='lower left')

    # plt.figure()
    # plt.hold('on')
    # for key in ['sensitivity', 'specificity']:
    #     plt.plot(results['thresholds'], results[key], label=key)
    #     plt.legend(loc='lower left')

    flag_save_fig = True

    if flag_save_fig:
        plt.ioff()

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.plot(
                results['specificity'], results['sensitivity'],
                label2style_dict[label], label=label)
    plt.axis((0, 1.05, 0, 1.05))
    plt.ylabel('sensitivity')
    plt.xlabel('specificity')
    plt.legend(loc='lower left', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    if flag_save_fig:
        plt.savefig(os.path.join(fig_dir, 'img_sens-spec' + FigExt))

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.plot(results['recall'], results['precision'],
                     label2style_dict[label], label=label)
    plt.axis((0, 1.05, 0, 1.05))
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.legend(loc='lower right', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    if flag_save_fig:
        plt.savefig(os.path.join(fig_dir, 'img_precision-recall' + FigExt))

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.semilogx(
                1 - results['thresholds_pr'], results['fscore'],
                label2style_dict[label], label=label)
    # plt.axis((0, 1.05, 0, 1.05))
    plt.ylabel('F-score')
    plt.xlabel('1 - threshold')
    plt.legend(loc='upper left', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    if flag_save_fig:
        plt.savefig(os.path.join(fig_dir, 'img_fscore' + FigExt))

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.semilogx(
                1 - results['thresholds'], results['mcc'],
                label2style_dict[label], label=label)
    # plt.axis((0, 1.05, 0, 1.05))
    plt.ylabel('Matthews correlation coefficient')
    plt.xlabel('1 - threshold')
    plt.legend(loc='upper left', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    if flag_save_fig:
        plt.savefig(os.path.join(fig_dir, 'img_mcc' + FigExt))

    if flag_save_fig:
        plt.close('all')
    else:
        plt.show()

    best_thresh_fscore = results['thresholds_pr'][np.argmax(results['fscore'])]
