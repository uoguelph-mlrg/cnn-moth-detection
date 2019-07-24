from pipeline import load_initial_data_wrap, proc_config
from fileop import loadfile
from bowsvm import generate_vocabulary, generate_bow_feature
from bowsvm import get_not_none_data, recover_image_from_vector
from bowsvm import predict_none_as_zero

import numpy as np
import sklearn.svm
from sklearn.metrics import accuracy_score


def temp_show_none_label(features, labels):
    num_total = 0
    num_neg = 0
    for ind in range(len(features)):
        if features[ind] is None:
            num_total += 1
            if labels[ind] == 0:
                num_neg += 1

    print 'number of None: {}, number of negative: {}'.format(
        num_total, num_neg)


def predict_with_none_as_0(features, predict_fcn):
    '''
    features: list
    '''
    not_none_features, not_none_indices = get_not_none_data(features)
    labels = np.zeros(len(features))
    labels[not_none_indices] = predict_fcn(not_none_features)
    return labels


@predict_none_as_zero
def predict_with_none_as_0_decorated(features):
    return svc.predict(features)


if __name__ == '__main__':
    config = loadfile('config.yaml')
    config = proc_config(config)

    X_train_init, y_train_init, X_valid_init, y_valid_init = \
        load_initial_data_wrap(config)

    train_img_list = recover_image_from_vector(X_train_init)
    valid_img_list = recover_image_from_vector(X_valid_init)

    voc = generate_vocabulary(train_img_list)

    total_train_features = generate_bow_feature(train_img_list, voc)
    temp_show_none_label(total_train_features, y_train_init)
    total_valid_features = generate_bow_feature(valid_img_list, voc)
    temp_show_none_label(total_valid_features, y_valid_init)

    train_features, train_indices = get_not_none_data(total_train_features)

    # kernel = 'rbf'
    c_const = 100.
    kernel = 'linear'

    svc = sklearn.svm.SVC(kernel=kernel, C=c_const)
    svc.fit(train_features, y_train_init[train_indices])

    labels_pred_train = predict_with_none_as_0(total_train_features,
                                               svc.predict)
    labels_pred_valid = predict_with_none_as_0(total_valid_features,
                                               svc.predict)
    print accuracy_score(y_train_init, labels_pred_train)
    print accuracy_score(y_valid_init, labels_pred_valid)

    print "Same results from a decorated function"
    labels_pred_train = predict_with_none_as_0(total_train_features,
                                               svc.predict)
    labels_pred_valid = predict_with_none_as_0(total_valid_features,
                                               svc.predict)
    print accuracy_score(y_train_init, labels_pred_train)
    print accuracy_score(y_valid_init, labels_pred_valid)
