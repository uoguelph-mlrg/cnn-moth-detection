import os
import sys
import time
import copy

import numpy as np

import theano
import theano.tensor as T
from tools_theano import shared_dataset
import logging


def fit(classifier, train_set,
        flag_report_valid=False, valid_set=None,
        flag_report_test=False, test_set=None,
        early_stop=True,
        flag_collect_weights=False,
        optimize_type=None,
        alpha_momentum=0.9,
        flag_minibatch_update=False,
        step_size=1):
    '''
    If flag_report_valid is false, early_stop and flag_report_test should be 
    automatically set to false
    if early_stop is false, then flag_report_test should be set to false

    to test fit, can run the mlp_class or convnet_class scripts


    flag_minibatch_update and step_size is for 
    '''

    # check if valid_set / test_set input is consistency with their flag

    if valid_set is None or None in valid_set:
        if flag_report_valid:
            logging.warn('Invalid validation set input!!!')
            flag_report_valid = False

    if test_set is None or None in test_set:
        if flag_report_test:
            logging.warn('Invalid test set input!!!')
            flag_report_test = False

    # suppress DeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    ########################
    # build training model #

    X, y = train_set
    if classifier.shuffle:
        idx = np.arange(X.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        train_set = (X, y)

    train_set_x, train_set_y = shared_dataset(train_set)
    if np.issubdtype(y.dtype, int):
        train_set_y = T.cast(train_set_y, 'int32')

    # compute number of minibatches for training
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= classifier.batch_size

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.

    givens = {classifier.x: train_set_x[classifier.index * classifier.batch_size: (classifier.index + 1) * classifier.batch_size],
              classifier.y: train_set_y[classifier.index * classifier.batch_size: (classifier.index + 1) * classifier.batch_size]}
    updates = []

    if optimize_type is None:
        # regular gradient descent with fixed learning rate

        for param_i, grad_i in zip(classifier.params, classifier.grads):
            updates.append(
                (param_i, param_i - classifier.learning_rate * grad_i))

        train_model = theano.function([classifier.index], classifier.cost,
                                      updates=updates,
                                      givens=givens)

    elif optimize_type == "momentum":
        pass
        # initialize a list of shared variables: velocities
        # each element in velocities corresponds to each element in
        # classifier.params and classifier.grads

        velocities = []

        for param_i in classifier.params:
            velocities.append(
                theano.shared(value=np.zeros(param_i.shape.eval(),
                                             dtype=param_i.dtype),
                              name=param_i.name,
                              borrow=True))

        # define updates_velo for updating velocity at each round
        updates_velo = []

        # update velocity then update weights
        for velo_i, grad_i in zip(velocities, classifier.grads):
            updates_velo.append(
                (velo_i, alpha_momentum * velo_i - classifier.learning_rate * grad_i))

        for param_i, velo_i in zip(classifier.params, velocities):
            updates.append((param_i, param_i + velo_i))

        train_momentum = theano.function([classifier.index], [],
                                         givens=givens,
                                         updates=updates_velo)
        train_model = theano.function([], [], updates=updates)

    elif optimize_type == "rmsprop":

        # a list of mean_squares correspond to different param_i in
        # classifier.params
        mean_squares = []

        for param_i in classifier.params:
            mean_squares.append(theano.shared(value=np.zeros(1, dtype=param_i.dtype),
                                              name=param_i.name, borrow=True))

        updates_ms = []

        coeff_avg = 0.9

        for ms_i, grad_i in zip(mean_squares, classifier.grads):
            updates_ms.append(
                (ms_i, coeff_avg * ms_i + (1. - coeff_avg) * T.sum(grad_i ** 2)))

        train_ms = theano.function([classifier.index], [],
                                   givens=givens,
                                   updates=updates_ms)

        mean_square_val = T.sqrt(sum(mean_squares))

        for param_i, grad_i in zip(classifier.params, classifier.grads):
            # updates.append((param_i, param_i - T.true_div(grad_i, mean_square_val)))
            updates.append((param_i, param_i - grad_i / mean_square_val[0]))

        train_model = theano.function([classifier.index], classifier.cost,
                                      givens=givens,
                                      updates=updates)

    else:
        raise NotImplementedError(
            "Error: optimize_type can only be None, momentum, rmsprop")
    # end if optimize_type

    # build training model #
    ########################

    ###############
    # train model #

    logging.info('... training')

    best_params = None
    best_valid_error = np.inf
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    # would be list of floats, elements are error rates in different rounds
    train_error_list = []
    valid_error_list = []
    train_obj_list = []
    valid_obj_list = []

    # this will be a list of list, each element is a list containing weights in
    # the same order as specified in classifier.params
    weights_list = []

    epoch = 0
    while (epoch < classifier.n_epochs):
        epoch = epoch + 1

        # train model over minibatches
        for minibatch_index in xrange(n_train_batches):
            # cost_ij = train_model(minibatch_index)

            if optimize_type is None:
                train_model(minibatch_index)

            elif optimize_type == "momentum":
                train_momentum(minibatch_index)
                train_model()

            elif optimize_type == "rmsprop":
                train_ms(minibatch_index)
                train_model(minibatch_index)

        # end for minibatch_index


        # compute zero-one loss on training set
        this_train_error = \
            np.mean(classifier.predict(X) != y)

        # compute

        # record objective function value
        this_train_obj = classifier.predict_cost(X, y)
        train_obj_list.append(this_train_obj)

        train_disp_str = 'ep %i, %i, t-err %.4f%%, t-obj %.2e' % (
            epoch, n_train_batches, this_train_error * 100., this_train_obj)

        train_error_list.append(this_train_error)

        if flag_collect_weights:
            weights = []
            # record the weights of current round
            for ind_params in range(len(classifier.params)):
                weights.append(classifier.params[ind_params].get_value())

            weights_list.append(weights)

        # skip reporting validation data score and early stopping
        if flag_report_valid:

            this_valid_obj = classifier.predict_cost(
                valid_set[0], valid_set[1])
            valid_obj_list.append(this_valid_obj)

            # compute zero-one loss on validation set
            this_valid_error = \
                np.mean(classifier.predict(valid_set[0]) != valid_set[1])

            valid_disp_str = ', v-err %.4f%%, v-obj %.2e' % (
                this_valid_error * 100., this_valid_obj)

            logging.info(train_disp_str + valid_disp_str)
            valid_error_list.append(this_valid_error)

        else:
            logging.info(train_disp_str)

        ##################
        # early stopping #

        # if we got the best validation score until now
        if early_stop and this_valid_error < best_valid_error:

            # save best validation score, parameters and iteration number
            best_valid_error = this_valid_error
            best_params = copy.deepcopy(classifier.params)
            best_epoch = epoch

            if flag_report_test:
                # test it on the test set
                test_score = \
                    np.mean(classifier.predict(test_set[0]) != test_set[1])

                test_disp_str = 'best: test err %.4f%%' % (test_score * 100.)

                logging.info(test_disp_str)

        # early stopping #
        ##################

        # end for (the last break controls this)

    # end while (done_looping controls this)

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at epoch %i,'
          'with test performance %f %%' %
          (best_valid_error * 100., best_epoch + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    fit_outputs_dict = {}
    fit_outputs_dict['best_params'] = best_params
    fit_outputs_dict['train_error_list'] = train_error_list
    fit_outputs_dict['valid_error_list'] = valid_error_list
    fit_outputs_dict['train_obj_list'] = train_obj_list
    fit_outputs_dict['valid_obj_list'] = valid_obj_list
    fit_outputs_dict['weights_list'] = weights_list

    return fit_outputs_dict
# end def fit


if __name__ == '__main__':
    from fileop import load_data
    from mlp_class import MLP
    from misc import set_quick_logging

    set_quick_logging()

    datasets = load_data(data_name='cifar100')

    clf = MLP(n_epochs=5, batch_size=200, n_in=3072, n_out=100)

    fit_dict = \
        fit(clf,
            train_set=datasets[0],
            valid_set=datasets[1],
            test_set=datasets[2],
            flag_report_test=True,
            flag_report_valid=True,
            early_stop=True,
            optimize_type="momentum")
