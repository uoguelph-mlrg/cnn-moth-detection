

from evaluation import get_perform_measures_from_file, get_avg_miss_rate, get_fscore
from pipeline import proc_config
from fileop import loadfile
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from config import FigExt


def get_performance_wrap(run_path, det_file, beta=2):

    config = loadfile(os.path.join(run_path, 'config.yaml'))
    config = proc_config(config)
    detections_test_file = os.path.join(run_path, det_file)

    thresh, fppi, miss_rate, recall, precision = \
        get_perform_measures_from_file(
            os.path.join(config['data_path'], config['detect_test_set']),
            detections_test_file,
            overlap=config['overlap_threshold'])

    fscore = get_fscore(recall, precision, beta=beta)
    precision_recall_auc = metrics.auc(
        recall, precision, reorder=True)

    logavg_miss_rate = get_avg_miss_rate(fppi, miss_rate)

    return dict(
        precision_recall_auc=precision_recall_auc,
        logavg_miss_rate=logavg_miss_rate,
        thresh=thresh,
        fppi=fppi,
        miss_rate=miss_rate,
        recall=recall,
        precision=precision,
        fscore=fscore,
    )


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
        'ConvNet 21': 'g-',
        'LogReg 35': 'b--',
    }

    font = {'family': 'normal',
            'size': 20}

    matplotlib.rc('font', **font)

    list_results = []
    for run_path, det_file in zip(list_run_path, list_det_file):
        list_results.append(get_performance_wrap(run_path, det_file, beta=2))

    for results, label in zip(list_results, list_labels):
        print "{}: precision-recall AUC: {:.3f}".format(
            label, results['precision_recall_auc'])
        print "{}: log-average miss rate: {:.3f}".format(
            label, results['logavg_miss_rate'])

    plt.ioff()

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.semilogx(results['fppi'], results['miss_rate'],
                         label2style_dict[label], label=label)

    plt.xlabel('false positive per image')
    plt.ylabel('miss rate')
    plt.legend(loc='upper right', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'mr-fppi' + FigExt))

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.plot(results['recall'], results['precision'],
                     label2style_dict[label], label=label)


    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.legend(loc='lower left', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    # plt.xlim((0., 1.))
    # plt.ylim(0., 1.)
    # plt.axis((0, 1, 0, 1))
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    # plt.axis('equal')
    # ax.set_xlim((0., 1.))
    # ax.set_ylim(0., 1.)
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'precision-recall' + FigExt))

    plt.figure()
    plt.hold('on')
    for results, label in zip(list_results, list_labels):
        if label in label2style_dict:
            plt.semilogx(1 - results['thresh'], results['fscore'],
                         label2style_dict[label], label=label)
    # plt.axis((0, 1.05, 0, 1.05))
    plt.ylabel('F-score')
    plt.xlabel('1 - threshold')
    plt.legend(loc='upper left', fontsize=font['size'],
               fancybox=True, framealpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fscore' + FigExt))

    best_thresh_fscore = results['thresh'][np.argmax(results['fscore'])]

    plt.close('all')
