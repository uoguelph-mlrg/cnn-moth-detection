import cv2
import numpy as np
import os
# from multiprocessing import Process
from sklearn.svm import SVC
# import matplotlib.pyplot as plt
import cPickle as pkl
from fileop import loadfile
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
# import ipdb


# ======== obsolete ========
# VVVVVVVVVVVVVVVVVVVVVVVVVV


def generate_vocabulary(img_list, num_cluster=128,
                        detector=None, extractor=None):
    if detector is None:
        detector = cv2.FeatureDetector_create('SIFT')
    if extractor is None:
        extractor = cv2.DescriptorExtractor_create('SIFT')

    bow_trainer = cv2.BOWKMeansTrainer(clusterCount=num_cluster)
    for img in img_list:
        temp = extractor.compute(img, detector.detect(img))[1]
        if temp is not None:
            bow_trainer.add(temp)

    voc = bow_trainer.cluster()
    return voc


def generate_bow_feature(img_list, voc,
                         detector=None, extractor=None, matcher=None):
    if detector is None:
        detector = cv2.FeatureDetector_create('SIFT')
    if extractor is None:
        extractor = cv2.DescriptorExtractor_create('SIFT')
    if matcher is None:
        matcher = cv2.DescriptorMatcher_create('BruteForce')

    bow_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    bow_extractor.setVocabulary(voc)

    feature_list = []
    for img in img_list:
        temp = bow_extractor.compute(img, detector.detect(img))
        if temp is not None:
            feature_list.append(temp[0])
        else:
            feature_list.append(None)

    return feature_list


# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ======== obsolete ========


def load_imgs_labels(img_pos_folder, img_neg_folder, img_ext='.jpg'):

    img_pos_path_list = [os.path.join(img_pos_folder, img_name)
                         for img_name in os.listdir(img_pos_folder)
                         if img_ext in img_name]
    img_neg_path_list = [os.path.join(img_neg_folder, img_name)
                         for img_name in os.listdir(img_neg_folder)
                         if img_ext in img_name]

    img_path_list = img_neg_path_list + img_pos_path_list

    imgs = [cv2.imread(img_path) for img_path in img_path_list]
    labels = np.hstack((np.zeros(len(img_neg_path_list)),
                        np.ones(len(img_pos_path_list))))

    return imgs, labels


def get_not_none_data(features):
    not_none_indices, not_none_features = zip(
        *[(ind, item) for ind, item in enumerate(features)
          if item is not None])
    return np.array(not_none_features), np.array(not_none_indices)


def predict_none_as_zero(predict_func):
    '''Intended to be used as decorator
    '''
    def func_wrapper(features):
        not_none_features, not_none_indices = get_not_none_data(features)
        labels = np.zeros(len(features))
        labels[not_none_indices] = predict_func(not_none_features)
        return labels
    return func_wrapper


def func_wrap(img, detector, extractor):
    img = img.astype('uint8')
    temp = extractor.compute(img, detector.detect(img))
    temp = None
    if temp is not None:
        return temp[0]
    else:
        return None


def func_generate_bow_feature(img_list,
                              voc,
                              detector_type='Dense',
                              extractor_type='SIFT',
                              matcher_type='BruteForce',
                              ):
    '''
    Function for parallel by multiprocessing
    '''
    print "func_generate_bow_feature detector_type {}".format(detector_type)

    detector = cv2.FeatureDetector_create(detector_type)
    extractor = cv2.DescriptorExtractor_create(extractor_type)
    matcher = cv2.DescriptorMatcher_create(matcher_type)

    bow_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    bow_extractor.setVocabulary(voc)


    feature_list = []
    for img in img_list:
        img = img.astype('uint8')
        temp = bow_extractor.compute(img, detector.detect(img))
        if temp is not None:
            feature_list.append(temp[0])
        else:
            feature_list.append(None)

    return feature_list


def func_generate_feature(img_list,
                          detector_type='Dense',
                          extractor_type='SIFT',
                          ):
    print "func_generate_feature detector_type {}".format(detector_type)

    detector = cv2.FeatureDetector_create(detector_type)
    extractor = cv2.DescriptorExtractor_create(extractor_type)

    feature_list = []
    for img in img_list:
        img = img.astype('uint8')
        feature_list.append(extractor.compute(
            img, detector.detect(img))[1])

    return feature_list


def func_predict_proba(features, svm_model):
    return 1. / (1. + np.exp(-svm_model.decision_function(features)))


def dist_img_list(img_list, num_jobs):
    num_per_list = len(img_list) // num_jobs
    assert num_per_list * num_jobs <= len(img_list)

    img_list_list = []
    for ind in range(num_jobs):
        if ind < num_jobs - 1:
            img_list_list.append(
                img_list[ind * num_per_list:(ind + 1) * num_per_list])
        else:
            img_list_list.append(img_list[ind * num_per_list:])
    return img_list_list


class BowSvm(object):

    '''
    SVM parameters
        kernel type
        degree for polynomial kernel
        C

    BOW parameters:
        feature type
        number of clusters
    '''

    def __init__(self, num_cluster=128,
                 matcher_type='BruteForce',
                 detector_type='Dense',
                 extractor_type='SIFT',
                 num_jobs=8,
                 **svm_kwargs):
        '''
        matcher_type
        detector_type
        extractor_type

        see sklearn.svm.SVC for svm_kwargs

        '''

        self.num_cluster = num_cluster
        self.detector_type = detector_type
        self.extractor_type = extractor_type
        self.matcher_type = matcher_type
        self.svm = SVC(**svm_kwargs)
        self.num_jobs = num_jobs

        self.build_cv2()

    def build_cv2(self):
        self.detector = cv2.FeatureDetector_create(self.detector_type)
        self.extractor = cv2.DescriptorExtractor_create(self.extractor_type)
        self.matcher = cv2.DescriptorMatcher_create(self.matcher_type)
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.extractor,
                                                           self.matcher)
        self.bow_trainer = cv2.BOWKMeansTrainer(clusterCount=self.num_cluster)
        if 'voc' in dir(self):
            self.bow_extractor.setVocabulary(self.voc)

    def generate_vocabulary(self, img_list):
        if self.num_jobs > 1:
            img_list_list = dist_img_list(img_list, self.num_jobs)
            feature_list_list = Parallel(n_jobs=self.num_jobs)(
                delayed(func_generate_feature)(
                    img_list,
                    detector_type=self.detector_type,
                    extractor_type=self.extractor_type,
                )
                for img_list in img_list_list)

            feature_list = [feature for feature_list in feature_list_list
                            for feature in feature_list]

        elif self.num_jobs == 1:
            feature_list = func_generate_feature(
                img_list,
                detector_type=self.detector_type,
                extractor_type=self.extractor_type,
            )

        for feature in feature_list:
            if feature is not None:
                self.bow_trainer.add(feature)

        print "generating vocabulary using {} raw features ...".format(
            len(feature_list))
        self.voc = self.bow_trainer.cluster()
        print "vocabulary generated ..."
        self.bow_extractor.setVocabulary(self.voc)

    def generate_bow_feature(self, img_list):

        if self.num_jobs > 1:

            img_list_list = dist_img_list(img_list, self.num_jobs)

            feature_list_list = Parallel(n_jobs=self.num_jobs)(
                delayed(func_generate_bow_feature)(
                    img_list, voc=self.voc,
                    detector_type=self.detector_type,
                    extractor_type=self.extractor_type,
                    matcher_type=self.matcher_type)
                for img_list in img_list_list)

            feature_list = [feature for feature_list in feature_list_list
                            for feature in feature_list]

        elif self.num_jobs == 1:
            feature_list = func_generate_bow_feature(
                img_list, voc=self.voc,
                detector_type=self.detector_type,
                extractor_type=self.extractor_type,
                matcher_type=self.matcher_type)

        else:
            raise ValueError("self.num_jobs={}, should be >= 1".format(
                self.num_jobs))

        return feature_list


    def fit(self, img_list, labels):
        print '111111111111111'
        self.generate_vocabulary(img_list)
        print '222222222222222'
        features = self.generate_bow_feature(img_list)
        print '333333333333333'
        not_none_features, not_none_indices = get_not_none_data(features)
        print 'start fitting svm...'
        print 'number of svm training examples {}'.format(
            len(not_none_features))
        self.svm.fit(not_none_features, labels[not_none_indices])

    def predict_proba(self, img_list):
        features = self.generate_bow_feature(img_list)
        not_none_features, not_none_indices = get_not_none_data(features)
        y_prob = np.zeros(len(img_list))

        if self.num_jobs == 1:
            y_prob_not_none = 1. / (1. + np.exp(
                -self.svm.decision_function(not_none_features)))

        else:
            num_per_list = len(not_none_features) // self.num_jobs
            assert num_per_list * self.num_jobs <= len(not_none_features)
            not_none_features_list = []
            for ind in range(self.num_jobs):
                if ind < self.num_jobs - 1:
                    not_none_features_list.append(np.array(
                        not_none_features[ind * num_per_list:
                                          (ind + 1) * num_per_list]))
                else:
                    not_none_features_list.append(np.array(
                        not_none_features[ind * num_per_list:]))

            y_prob_not_none_list = Parallel(n_jobs=self.num_jobs)(
                delayed(func_predict_proba)(
                    not_none_features, svm_model=self.svm)
                for not_none_features in not_none_features_list)

            y_prob_not_none = np.concatenate(y_prob_not_none_list, axis=0)


        y_prob[not_none_indices] = y_prob_not_none
        y_prob = np.c_[1 - y_prob, y_prob]
        return y_prob

    def predict(self, img_list):
        proba = self.predict_proba(img_list)
        return np.argmax(proba, axis=1)

    def __getstate__(self):
        print '__getstate__ executed'
        list_to_del = ['detector', 'extractor', 'matcher',
                       'bow_extractor', 'bow_trainer']
        state = self.__dict__.copy()
        for key in state.keys():
            if key in list_to_del:
                del state[key]
        return state

    def __setstate__(self, state):
        print '__setstate__ executed'
        self.__dict__ = state
        self.build_cv2()


def recover_image_from_vector(data, shape=(-1, 3, 28, 28), axis=1, start=4):
    return np.rollaxis(data.reshape(shape), axis, start).mean(
        axis=-1).astype('uint8')


def test_on_whole_image(flag_toy=True):
    root_folder = os.path.expanduser(
        '~/Data/bugs_annotated_2014/new_separation')

    if flag_toy:
        train_pos_folder = os.path.join(root_folder, 'train_toy/withmoth/')
        train_neg_folder = os.path.join(root_folder, 'train_toy/nomoth/')
        valid_pos_folder = os.path.join(root_folder, 'valid_toy/withmoth/')
        valid_neg_folder = os.path.join(root_folder, 'valid_toy/nomoth/')
        test_pos_folder = os.path.join(root_folder, 'test_toy/withmoth/')
        test_neg_folder = os.path.join(root_folder, 'test_toy/nomoth/')
    else:
        train_pos_folder = os.path.join(root_folder, 'train/withmoth/')
        train_neg_folder = os.path.join(root_folder, 'train/nomoth/')
        valid_pos_folder = os.path.join(root_folder, 'valid/withmoth/')
        valid_neg_folder = os.path.join(root_folder, 'valid/nomoth/')
        test_pos_folder = os.path.join(root_folder, 'test/withmoth/')
        test_neg_folder = os.path.join(root_folder, 'test/nomoth/')

    num_cluster = 8

    train_img_list, train_labels = load_imgs_labels(train_pos_folder,
                                                    train_neg_folder)
    valid_img_list, valid_labels = load_imgs_labels(valid_pos_folder,
                                                    valid_neg_folder)
    test_img_list, test_labels = load_imgs_labels(test_pos_folder,
                                                  test_neg_folder)


    bowsvm = BowSvm(num_cluster=num_cluster)
    bowsvm.fit(train_img_list, train_labels)
    bowsvm.predict_proba(valid_img_list)
    bowsvm.predict_proba(test_img_list)


def test_on_patches():

    config = loadfile('config.yaml')
    config['flag_debug'] = True
    config = proc_config(config)


    X_train_init, y_train_init, X_valid_init, y_valid_init =  \
        load_initial_data_wrap(config)

    train_img_list = recover_image_from_vector(X_train_init)
    valid_img_list = recover_image_from_vector(X_valid_init)

    bowsvm = BowSvm(kernel='linear', C=100.)

    time_start = time.time()
    print 'training...'
    bowsvm.fit(train_img_list, y_train_init)
    print time.time() - time_start

    time_start = time.time()
    labels_pred_train = bowsvm.predict(train_img_list)
    labels_pred_valid = bowsvm.predict(valid_img_list)
    print time.time() - time_start
    print accuracy_score(y_train_init, labels_pred_train)
    print accuracy_score(y_valid_init, labels_pred_valid)


if __name__ == '__main__':
    from pipeline import load_initial_data_wrap, proc_config
    import time

    # test pickle
    svm_0 = BowSvm()
    with open('svm.pkl', 'w') as f:
        pkl.dump(svm_0, f)

    with open('svm.pkl', 'r') as f:
        svm_1 = pkl.load(f)

    test_on_patches()
    # test_on_whole_image()
