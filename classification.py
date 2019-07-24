import os
import yaml
import logging
import cPickle as pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import preprocessing, metrics

from fit import fit
from convnet_class import lenet5
from bowsvm import BowSvm, recover_image_from_vector
from tools import rot_img_array
from tools_preproc import preprocessor
from data_munging import get_pos, get_neg

logger = logging.getLogger(__name__)


def combine_pos_neg(input_pos, input_neg,
                    flag_rot_aug=True,
                    flag_flatten=True,
                    ):
    '''
    Combine wrongly classified positive examples and 
    negative examples

    designed to work for both gray and color image
    if color input, the dimension of each single element 
    should be 3*28*28

    input_pos or input_neg can be both list and ndarray
    for grey image patches, it should have dimension like
      [n_example, n_row, n_col]
    for color patches, it should have
      [n_example, n_color, n_row, n_col]

    proprocessing patch lists to be patch ndarray
    '''

    input_neg = patch_preproc(input_neg, flag_rot_aug=flag_rot_aug)

    # dealing with the situation of no positive examples
    if len(input_pos):
        input_pos = patch_preproc(input_pos, flag_rot_aug=flag_rot_aug)
        X = np.cast[np.float32](np.r_[np.asarray(input_pos),
                                      np.asarray(input_neg)])
    else:
        X = np.cast[np.float32](np.asarray(input_neg))

    # resize does not work for non single-element array
    # X.resize((X.shape[0], np.prod(X.shape[1:])))

    if flag_flatten:
        X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))

    y = np.cast[np.int32](np.r_[np.ones(len(input_pos)),
                                np.zeros(len(input_neg))])

    return X, y
# end def combine_pos_neg


def patch_preproc(patch_list, flag_rot_aug=True):
    '''
    perform rotational data augmentation

    flag_rot_aug, whether do rotation augmentation on data
    '''
    # each element in patch_list is either 28 * 28 * 3 or 28 * 28
    patch_array = np.asarray(patch_list)

    # if has 4 dimensions, means it's color image
    if patch_array.ndim == 4 and patch_array.shape[3] == 3:
        patch_array = np.rollaxis(patch_array, 3, 1)
    # now patch_array is n_example * 3 * 28 * 28 or n_example * 28 * 28

    # do data augmentation
    #
    # rotation
    if flag_rot_aug:
        patch_array_list = []

        for ind_rot in range(8):
            patch_array_list.append(rot_img_array(patch_array, kind=ind_rot))

        patch_array = np.concatenate(patch_array_list, axis=0)
        # now patch_array is (8 * n_example) * 3 * 28 * 28
        # or (8 * n_example) * 28 * 28

    return patch_array
# end def patch_preproc


def load_initial_data(data_path, target_width, target_height,
                      flag_rescale=False, flag_take_valid=False,
                      train_subpath_pos='train/withmoth',
                      train_subpath_neg='train/nomoth',
                      valid_subpath_pos='valid/withmoth',
                      valid_subpath_neg='valid/nomoth',
                      flag_multiscale=False,
                      detect_width_list=[8, 16, 32, 64],
                      detect_height_list=[8, 16, 32, 64],
                      flag_rgb=True,
                      flag_rot_aug_train=True,
                      flag_rot_aug_valid=True,
                      flag_trans_aug_train=True,
                      flag_trans_aug_valid=True,
                      dist_trans_list=(-2, 0, 2),
                      ):

    # input:
    #   data_path, flag_rescale, target_width, target_height
    #   use_all_falsepositives, ind_retrain, write_path
    # output: X, y, X_valid, y_valid

    train_path_pos = os.path.join(data_path, train_subpath_pos)
    train_path_neg = os.path.join(data_path, train_subpath_neg)

    train_pos = get_pos(train_path_pos, target_height, target_width,
                        flag_rescale, flag_multiscale=flag_multiscale,
                        flag_rgb=flag_rgb,
                        detect_width_list=detect_width_list,
                        detect_height_list=detect_height_list,
                        flag_trans_aug=flag_trans_aug_train,
                        dist_trans_list=dist_trans_list,
                        )

    train_neg = get_neg(train_path_neg,
                        target_height, target_width,
                        flag_rescale, flag_rgb=flag_rgb,
                        num_appr=len(train_pos),
                        flag_trans_aug=flag_trans_aug_train,
                        dist_trans_list=dist_trans_list,
                        )

    X, y = combine_pos_neg(train_pos, train_neg,
                           flag_rot_aug=flag_rot_aug_train)

    if flag_take_valid:
        valid_path_pos = os.path.join(data_path, valid_subpath_pos)
        valid_path_neg = os.path.join(data_path, valid_subpath_neg)

        valid_pos = get_pos(valid_path_pos, target_height, target_width,
                            flag_rescale, flag_multiscale=flag_multiscale,
                            flag_rgb=flag_rgb,
                            detect_width_list=detect_width_list,
                            detect_height_list=detect_height_list,
                            flag_trans_aug=flag_trans_aug_valid,
                            dist_trans_list=dist_trans_list,
                            )

        valid_neg = get_neg(valid_path_neg,
                            target_height, target_width,
                            flag_rescale, flag_rgb=flag_rgb,
                            num_appr=len(valid_pos),
                            flag_trans_aug=flag_trans_aug_valid,
                            dist_trans_list=dist_trans_list,
                            )

        X_valid, y_valid = combine_pos_neg(valid_pos, valid_neg,
                                           flag_rot_aug=flag_rot_aug_valid)

    else:
        X_valid, y_valid = None, None

    return X, y, X_valid, y_valid

# end def load_initial_data


def load_misclf_data(write_path, ind_train=0,
                     use_all_falsepositives=True, flag_rgb=True,
                     flag_rot_aug=True, data_set='train'):
    '''
    Load misclassified data previous
    '''

    X_more, y_more = None, None

    if data_set == 'train':
        file_name = 'misdetects'
    elif data_set == 'valid':
        file_name = 'misdetects_valid'
    else:
        raise NotImplementedError("data_set should be either train or test")

    # This try catch part is for retraining, therefore wrongly classified
    # examples are updated
    # check for updated wrongly classified examples
    # Usually the pos_examples are repetitions of original training data, and
    # neg_examples are newly generated from false positive detections_file
    try:
        import cPickle as pickle
        with open(os.path.join(write_path, file_name + '.pkl'), 'rb') as f:
            pos_examples, neg_examples = pickle.load(f)

            X_more, y_more = combine_pos_neg(pos_examples, neg_examples,
                                             flag_rot_aug=flag_rot_aug)
            logger.debug(data_set +
                         ": added last round misclassified examples")

        # add all previous misdetects if needed
        if use_all_falsepositives:
            for i in range(ind_train - 1):
                with open(os.path.join(write_path,
                                       file_name + str(i) + '.pkl'),
                          'rb') as f:

                    pos_examples, neg_examples = pickle.load(f)

                    X_more, y_more = \
                        combine_pos_neg(pos_examples, neg_examples,
                                        flag_rot_aug=flag_rot_aug)

                    logger.debug(data_set + ": added round" +
                                 str(i) + "misclassified examples")

    except IOError:
        logger.debug("not loading additional negative examples")

    return X_more, y_more
# end def load_misclf_data


class SVM(object):

    '''
    SVM here only works for 2 class problem
    '''

    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def predict_proba(self, *args, **kwargs):
        y_prob = self.model.decision_function(*args, **kwargs)
        y_prob = 1. / (1. + np.exp(y_prob))
        y_prob = np.c_[1 - y_prob[:, 0], y_prob[:, 0]]
        return y_prob
        # return self.model.predict_proba(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


class NullScaler(object):

    def transform(self, x):
        return x


def train_core(X, y, X_valid=None, y_valid=None, n_iter=500,
               target_width=32, target_height=32, class_weight=None,
               flag_rgb=True,
               classifier_type="convnet",
               preproc_type="univar",
               bowsvm_config=None,
               ):
    '''
    core function for training, including data preprocessing and classifier
    training
    '''

    # data preprocessing #
    # input/output: X, X_valid

    if preproc_type == 'none':
        scaler = NullScaler()
        if classifier_type == 'bowsvm':
            scaler.transform = recover_image_from_vector
        X = scaler.transform(X)
        X_valid = scaler.transform(X_valid)

    elif preproc_type == "univar":
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        X_valid = scaler.transform(X_valid)
    elif preproc_type == "zca":
        scaler = preprocessor(flag_zca=True)
        X = scaler.fit(X)
        X_valid = scaler.fit(X_valid)
    elif preproc_type == "lcn":
        scaler = preprocessor(flag_lcn=True)
        X = scaler.fit(X)
        X_valid = scaler.fit(X_valid)
    else:
        raise NotImplementedError(
            'Currently only implemented preproc_type: univar, zca and lcn.')

    # need to make y int

    # train classifier #
    # input: X, y
    # output: clf

    # here clf combines log loss and l2 penalty, which is equivalent to logistic regression
    # alpha is the coeff for l2 qregularization term

    if classifier_type == "convnet":
        print X.shape[0]

        if flag_rgb:
            img_dim = 3
        else:
            img_dim = 1

        clf = lenet5(n_epochs=n_iter, batch_size=256,
                     learning_rate=0.002, img_dim=img_dim,
                     nkerns=[32, 48], num_class=2)

        fit_outputs = fit(clf, train_set=(X, y),
                          flag_report_test=False,
                          flag_report_valid=True,
                          early_stop=True,
                          valid_set=(X_valid, y_valid),
                          optimize_type='momentum')

        clf.set_weights(fit_outputs['best_params'])

    elif classifier_type == "logreg":

        clf = SGDClassifier(loss="log", penalty="l2",
                            n_iter=n_iter, verbose=True,
                            shuffle=True, alpha=0.01,
                            class_weight=class_weight)
        clf.fit(X, y)

    elif classifier_type == "svm":

        clf = SVM(probability=True, verbose=True, kernel='linear')
        clf.fit(X, y)

    elif classifier_type == "bowsvm":

        clf = BowSvm(
            num_cluster=bowsvm_config['num_cluster'],
            verbose=True,
            kernel=bowsvm_config['kernel'],
            degree=bowsvm_config['degree'],
        )
        clf.fit(X, y)

    else:
        raise NotImplementedError(
            "classifier_type can only be convnet / logreg / svm / bow-svm")

    # report number of examples
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    n_train = y.shape[0]
    logger.info('training set: %d positive examples, %d negative examples, %d total'
                % (n_pos, n_neg, n_train))

    if y_valid is not None:
        n_pos = (y_valid == 1).sum()
        n_neg = (y_valid == 0).sum()
        n_train = y_valid.shape[0]
        logger.info('validation set: %d positive examples, %d negative examples, %d total'
                    % (n_pos, n_neg, n_train))

    return clf, scaler
# end def train_core


# train_simple_classifer is NOT MAINTAINED for recent updates
# fitting the classifier
# currently not really used in the whole pipeline
def train_simple_classifier(data_path, write_path,
                            target_width=32, target_height=32,
                            n_iter=500, class_weight=None,
                            use_all_falsepositives=True,
                            ind_train=0,
                            flag_rescale=True,
                            flag_multiscale=False,
                            flag_rgb=True,
                            detect_width_list=[8, 16, 32, 64],
                            detect_height_list=[8, 16, 32, 64]):

    # load initial data
    X, y, X_valid, y_valid = load_initial_data(
        data_path=data_path,
        target_width=target_width,
        target_height=target_height,
        flag_rescale=flag_rescale,
        flag_take_valid=True,
        flag_rgb=flag_rgb,
        flag_multiscale=flag_multiscale,
        detect_width_list=detect_width_list,
        detect_height_list=detect_height_list)

    # load misclassified data
    X_more, y_more = load_misclf_data(
        write_path, ind_train=ind_train,
        use_all_falsepositives=use_all_falsepositives,
        flag_rgb=flag_rgb)

    if X_more is not None:
        X = np.cast[np.float32](np.r_[X, X_more])
        y = np.cast[np.int32](np.r_[y, y_more])

    # train the classifier
    clf, scaler = train_core(
        X, y, X_valid=X_valid, y_valid=y_valid,
        n_iter=n_iter,
        target_width=target_width,
        target_height=target_height,
        class_weight=class_weight,
        flag_rgb=flag_rgb)

    # show performance
    y_pred = clf.predict(X)
    X_valid = scaler.transform(X_valid)
    y_pred_t = clf.predict(X_valid)

    logger.info("t_acc: %6.4f, v_acc: %6.4f" % (
        metrics.accuracy_score(y, y_pred),
        metrics.accuracy_score(y_valid, y_pred_t)))

    print "t_acc: %6.4f, v_acc: %6.4f" % (
        metrics.accuracy_score(y, y_pred),
        metrics.accuracy_score(y_valid, y_pred_t))

    return clf, scaler
# end def train_simple_classifier


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f)

    clf, scaler = train_simple_classifier(
        data_path=config['data_path'],
        write_path=config['write_path'],
        target_width=config['target_width'],
        target_height=config['target_height'],
        n_iter=config['n_iter'],
        class_weight=config['class_weight'])

    with open(os.path.join(config['write_path'],
                           'clf.pkl'), 'wb') as f:
        pickle.dump([clf, scaler], f)

    with open(os.path.join(config['write_path'],
                           'clf.pkl'), 'rb') as f:
        clf, scaler = pickle.load(f)
