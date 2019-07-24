import os
import shutil
import logging
import datetime
import argparse


import numpy as np
from sklearn import metrics

from fileop import loadfile, savefile
from evaluation import get_perform_measures_from_file
from evaluation import evaluate, get_avg_miss_rate
from classification import load_initial_data, load_misclf_data
from classification import train_core
from detection import detect, detect_multiscale

# define global variables
__traindet__ = 'detections.pkl'
__validdet__ = 'detections_valid.pkl'
__testdet__ = 'detections_test.pkl'
__trainmisdet__ = 'misdetects.pkl'
__validmisdet__ = 'misdetects_valid.pkl'
__clf__ = 'clf.pkl'
__results__ = 'results.pkl'


# ## define local utility functions ###


def add_file_index(filename_full, ind_retrain):
    '''
    Rename .pkl files to have the round index after using.
    For example, in round i, it'll rename detections.pkl to
    detections{i-1}.pkl, because it was generated in the
    last round.

    '''
    if ind_retrain == 0:
        return

    filename, fileext = os.path.splitext(filename_full)
    os.rename(filename_full,
              filename + str(ind_retrain - 1) + fileext)
    return
# end def add_file_index


def get_folder_name(prefix='bug_run_'):
    '''
    Generate a folder name to save the files for this training.
    Everything involved with this run should be saved in
    this folder.
    '''
    datestamp = datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S')
    folder_name = prefix + datestamp + '_' + \
        str(np.random.randint(low=10, high=99))

    return folder_name


def proc_config(config, write_path=None, data_path=None):
    '''
    preprocessing configurations loaded from config.yaml
    '''
    # check if in debug mode
    if config['flag_debug']:
        logging.info(
            "Under debugging mode, using _toy parameters")
        for key in config:
            if key[-4:] == '_toy':
                config[key[:-4]] = config[key]

    # deal with the numpy values in config.yaml
    if isinstance(config['dist_trans_list'], basestring):
        # if is string assign values
        exec("""config['dist_trans_list'] = """ +
             config['dist_trans_list'])

    if write_path is None:
        write_path = get_folder_name()
        # overwrite the original write path by the path of this run
        # so now the write_path in config.yaml acts like a root path
        config['write_path'] = os.path.join(
            config['write_path'], write_path)
    else:
        config['write_path'] = write_path

    if data_path is not None:
        config['data_path'] = data_path

    config['data_path'] = os.path.expanduser(config['data_path'])
    config['write_path'] = os.path.expanduser(config['write_path'])

    return config


def load_initial_data_wrap(config):

    return load_initial_data(
        data_path=config['data_path'],
        target_width=config['target_width'],
        target_height=config['target_height'],
        flag_rescale=config['flag_rescale'],
        flag_multiscale=config['flag_multiscale'],
        train_subpath_pos=config['train_subpath_pos'],
        train_subpath_neg=config['train_subpath_neg'],
        valid_subpath_pos=config['valid_subpath_pos'],
        valid_subpath_neg=config['valid_subpath_neg'],
        detect_width_list=config['detect_width_list'],
        detect_height_list=config['detect_height_list'],
        flag_take_valid=config['flag_take_valid'],
        flag_rgb=config['flag_rgb'],
        flag_rot_aug_train=config['flag_rot_aug'],
        flag_rot_aug_valid=config['flag_rot_aug'],
        flag_trans_aug_train=config['flag_trans_aug'],
        flag_trans_aug_valid=config['flag_trans_aug'],
        dist_trans_list=config['dist_trans_list'],
    )


def load_misclf_data_wrap(config, ind_round, data_set):
    return load_misclf_data(
        write_path=config['write_path'],
        ind_train=ind_round,
        use_all_falsepositives=config['use_all_falsepositives'],
        flag_rgb=config['flag_rgb'],
        flag_rot_aug=config['flag_rot_aug'],
        data_set=data_set,
    )


def train_core_wrap(X, y, X_valid, y_valid, config):
    return train_core(X, y, X_valid=X_valid, y_valid=y_valid,
                      n_iter=config['n_iter'],
                      target_width=config['target_width'],
                      target_height=config['target_height'],
                      class_weight=config['class_weight'],
                      flag_rgb=config['flag_rgb'],
                      classifier_type=config['classifier_type'],
                      preproc_type=config['preproc_type'],
                      bowsvm_config=config['bowsvm_config'],
                      )


def detect_multiscale_wrap(config, thresh, dataset):

    if dataset == 'train':
        sub_path = config['detect_train_set']
    elif dataset == 'valid':
        sub_path = config['detect_valid_set']
    elif dataset == 'test':
        sub_path = config['detect_test_set']

    return detect_multiscale(
        data_path=os.path.join(config['data_path'], sub_path),
        write_path=config['write_path'],
        target_width=config['target_width'],
        target_height=config['target_height'],
        x_stride=config['target_stride_x'],
        y_stride=config['target_stride_y'],
        detect_width_list=config['detect_width_list'],
        detect_height_list=config['detect_height_list'],
        thresh=thresh,
        n_images=-1,
        flag_rgb=config['flag_rgb'],
        flag_usemask=config['flag_usemask'],
        thresh_mask=config['thresh_mask'],
        nms_thresh=config['nms_threshhold'],
        flag_det_rot_aug=config['flag_det_rot_aug'])


def detect_singlescale_wrap(config, thresh, dataset):

    if dataset == 'train':
        sub_path = config['detect_train_set']
    elif dataset == 'valid':
        sub_path = config['detect_valid_set']
    elif dataset == 'test':
        sub_path = config['detect_test_set']

    return detect(
        data_path=os.path.join(config['data_path'], sub_path),
        write_path=config['write_path'],
        thresh=thresh,
        target_width=config['target_width'],
        target_height=config['target_height'],
        x_stride=config['target_stride_x'],
        y_stride=config['target_stride_y'],
        n_images=-1,
        flag_rgb=config['flag_rgb'],
        flag_usemask=config['flag_usemask'],
        thresh_mask=config['thresh_mask'],
        nms_thresh=config['nms_threshhold'],
        flag_det_rot_aug=config['flag_det_rot_aug'])


def evaluate_wrap(config, dataset, ind_round=0,
                  prob_threshold=0., num_neg_to_pick=0):

    if dataset == 'train':
        sub_path = config['detect_train_set']
        det_file = __traindet__
    elif dataset == 'valid':
        sub_path = config['detect_valid_set']
        det_file = __validdet__
    elif dataset == 'test':
        sub_path = config['detect_test_set']
        det_file = __testdet__

    return evaluate(
        data_path=os.path.join(config['data_path'], sub_path),
        filename=os.path.join(config['write_path'], det_file),
        write_path=config['write_path'],
        num_neg_to_pick=num_neg_to_pick,
        prob_threshold=prob_threshold,
        overlap_threshold=config['overlap_threshold'],
        flag_rescale=config['flag_rescale'],
        target_width=config['target_width'],
        target_height=config['target_height'],
        flag_rgb=config['flag_rgb'],
        flag_trans_aug=config['flag_trans_aug'],
        dist_trans_list=config['dist_trans_list'],
        data_set_name=dataset,
        ind_round=ind_round
    )


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='data_path')
    parser.add_argument('-w', dest='write_path')
    args = parser.parse_args()

    # train classifier
    config = loadfile('config.yaml')
    config = proc_config(config, data_path=args.data_path,
                         write_path=args.write_path)

    run_path = config['write_path']
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    shutil.copyfile('config.yaml', os.path.join(
        run_path, 'config.yaml'))

    # specify file names used for saving parameters
    clf_file = os.path.join(config['write_path'], __clf__)
    detections_file = os.path.join(
        config['write_path'], __traindet__)
    detections_valid_file = os.path.join(
        config['write_path'], __validdet__)
    detections_test_file = os.path.join(
        config['write_path'], __testdet__)
    misdetects_file = os.path.join(
        config['write_path'], __trainmisdet__)
    misdetects_valid_file = os.path.join(
        config['write_path'], __validmisdet__)

    # ############# training-retraining rounds ##################

    # train, detect, evaluate and save mistakes for retraining
    # evaluate, and save mistakes for re-training
    n_retrain = config['n_retrain']

    X_train_init, y_train_init, X_valid_init, y_valid_init = \
        load_initial_data_wrap(config)

    if config['num_neg_to_pick']:
        num_neg_to_pick_train = config['num_neg_to_pick']
        num_neg_to_pick_valid = int(
            1. * X_valid_init.shape[0] /
            X_train_init.shape[0] * num_neg_to_pick_train)

        print '{} train negative patches to collect each round'.format(
            num_neg_to_pick_train)
        print '{} valid negative patches to collect each round'.format(
            num_neg_to_pick_valid)
    else:
        num_neg_to_pick_train = 0
        num_neg_to_pick_valid = 0

    for ind_round in xrange(n_retrain + 1):
        logger.info('training round %d' % ind_round)

        # load misclassified data
        X_train_more, y_train_more = \
            load_misclf_data_wrap(config, ind_round,
                                  data_set='train')
        X_valid_more, y_valid_more = \
            load_misclf_data_wrap(config, ind_round,
                                  data_set='valid')

        if X_train_more is not None:
            X_train = np.cast[np.float32](
                np.r_[X_train_init, X_train_more])
            y_train = np.cast[np.int32](
                np.r_[y_train_init, y_train_more])
        else:
            X_train, y_train = X_train_init, y_train_init
            print "No additional training data loaded!"

        if X_valid_more is not None:
            X_valid = np.cast[np.float32](
                np.r_[X_valid_init, X_valid_more])
            y_valid = np.cast[np.int32](
                np.r_[y_valid_init, y_valid_more])
        else:
            X_valid, y_valid = X_valid_init, y_valid_init
            print "No additional training data loaded!"

        if config['flag_randperm']:
            indices_perm = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices_perm]
            y_train = y_train[indices_perm]

        # train the classifier
        clf, scaler = train_core_wrap(
            X_train, y_train, X_valid, y_valid, config)

        # rename previous file and write classifier to disk
        add_file_index(clf_file, ind_round)
        savefile(clf_file, clf, scaler)

        # detection with threshold zero on validation data
        if config['flag_multiscale']:
            detections_valid = detect_multiscale_wrap(
                config, 0.0, 'valid')
            detections = detect_multiscale_wrap(
                config, 0.0, 'train')
            detections_test = detect_multiscale_wrap(
                config, 0.0, 'test')
        else:
            detections = detect_singlescale_wrap(config, 0.0, 'train')
            detections_valid = detect_singlescale_wrap(
                config, 0.0, 'valid')
            detections_test = detect_singlescale_wrap(
                config, 0.0, 'test')

        # rename previous file
        add_file_index(detections_file, ind_round)
        add_file_index(detections_valid_file, ind_round)
        add_file_index(detections_test_file, ind_round)

        # save the current round parameters
        savefile(detections_file, detections)
        savefile(detections_valid_file, detections_valid)
        savefile(detections_test_file, detections_test)

        # performance evaluation
        array_thresh, array_fppi, array_miss_rate, array_recall, \
            array_precision = get_perform_measures_from_file(
                os.path.join(config['data_path'],
                             config['detect_train_set']),
                detections_file,
                overlap=config['overlap_threshold'])

        array_thresh, array_fppi, array_miss_rate, array_recall, \
            array_precision = get_perform_measures_from_file(
                os.path.join(config['data_path'],
                             config['detect_valid_set']),
                detections_valid_file,
                overlap=config['overlap_threshold'])

        array_thresh, array_fppi, array_miss_rate, array_recall, \
            array_precision = get_perform_measures_from_file(
                os.path.join(config['data_path'],
                             config['detect_test_set']),
                detections_test_file,
                overlap=config['overlap_threshold'])

        print "precision-recall AUC: {}".format(
            metrics.auc(array_recall, array_precision,
                        reorder=True))
        print "log-average miss rate: {}".format(
            get_avg_miss_rate(array_fppi, array_miss_rate))

        pos_examples, neg_examples = evaluate_wrap(
            config, 'train', ind_round, 0.5,
            num_neg_to_pick_train)

        logger.info('saving misdetections to disk: %s' %
                    misdetects_file)
        add_file_index(misdetects_file, ind_round)

        if not config['flag_use_false_neg']:
            pos_examples = []

        savefile(misdetects_file, pos_examples, neg_examples)

        pos_examples, neg_examples = evaluate_wrap(
            config, 'valid', ind_round, 0.5,
            num_neg_to_pick_valid)
        # print len(pos_examples)
        logger.info('saving misdetections to disk: %s' %
                    misdetects_valid_file)
        add_file_index(misdetects_valid_file, ind_round)

        if not config['flag_use_false_neg']:
            pos_examples = []

        savefile(
            misdetects_valid_file, pos_examples, neg_examples)

        evaluate_wrap(config, 'test', ind_round, 0)

    # end for ind_round in xrange(n_retrain + 1)

    # change the name of final parameter files
    add_file_index(clf_file, n_retrain + 1)
    add_file_index(detections_file, n_retrain + 1)
    add_file_index(detections_valid_file, n_retrain + 1)
    add_file_index(detections_test_file, n_retrain + 1)
    add_file_index(misdetects_file, n_retrain + 1)
    add_file_index(misdetects_valid_file, n_retrain + 1)
