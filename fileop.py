'''
Provide gwd_tools for file operations.
'''
import cPickle
import os
import yaml
import gzip
import logging
import numpy as np


def savefile(filename, *args, **kwargs):
    '''
    todo: write comments

    if only args, 
        if multiple, save a tuple
        if single, save that single variable
    if only kwargs, save a dict
    if both, save a tuple (tuple, dict)

    note that savefile(filename, (a, b, c)) and savefile(filename, a, b, c) will be the same

    '''

    # determine how to save depending on existence of args and kwargs
    if len(args):

        # if only one input args, just save it without making it a tuple
        if len(args) == 1:
            args = args[0]

        if not len(kwargs):
            tosave = args
        else:
            tosave = (args, kwargs)
    else:
        if len(kwargs):
            tosave = kwargs
        else:
            print "You forget to save some variables!!!"
            raise 

    with open(filename, 'wb') as f:
        cPickle.dump(tosave, f)
# end def savefile



def loadfile(filename, ext=None):
    '''
    todo: write comments
    currently support yaml and pkl and their .gz file

    '''

    file_root, file_ext = os.path.splitext(filename)

    # the following line is good if no file name is involved
    # open_fcn = gzip.open if file_ext == '.gz' else open
    
    if file_ext == '.gz':
        _, file_ext = os.path.splitext(file_root)
        open_fcn = gzip.open
    else:
        open_fcn = open

    with open_fcn(filename, 'rb') as f:
        if file_ext == '.pkl' or ext == '.pkl':
            return cPickle.load(f)
        elif file_ext == '.yaml' or ext == '.yaml':
            return yaml.load(f)
        else:
            raise NotImplementedError('Currently only support .pkl and .yaml files')
# end def loadfile


def load_data(data_name='mnist'):
    ''' 
    adapted from the theano tutorial version, now is able to load mnist, cifar10 and cifar100

    The downloading part of the orignal code was removed

    currently the output is 

    Loads the data_name

    :type data_name: string
    :param data_name: the path to the data_name (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    logging.info('... loading data')

    if data_name.lower() == 'mnist':
        filename = os.path.join(os.environ['MNIST_PATH'], 'mnist.pkl.gz')
        train_set, valid_set, test_set = loadfile(filename)

    elif data_name.lower() == 'cifar10':
        data_path = os.path.join(os.environ['CIFAR10_PATH'], 
                                 'cifar-10-batches-py')
        X_train_list = []
        y_train_list = []
        for batch in range(1, 5):
            train_dict = loadfile(os.path.join(data_path, 'data_batch_1'))
        
            X_train_list.append(train_dict['data'])
            y_train_list.append(train_dict['labels'])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        valid_dict = loadfile(os.path.join(data_path, 'data_batch_5'))
        X_valid = valid_dict['data']
        y_valid = np.array(valid_dict['labels'])

        test_dict = loadfile(os.path.join(data_path, 'test_batch'))
        X_test = test_dict['data']
        y_test = np.array(test_dict['labels'])

        train_set = (X_train, y_train)
        valid_set = (X_valid, y_valid)
        test_set = (X_test, y_test)


    elif data_name.lower() == 'cifar100':
        data_path = os.path.join(os.environ['CIFAR100_PATH'], 
                                 'cifar-100-python') 
        train_dict = loadfile(os.path.join(data_path, 'train'))

        X_train = train_dict['data'][:40000]
        y_train = np.array(train_dict['fine_labels'][:40000])
        X_valid = train_dict['data'][40000:]
        y_valid = np.array(train_dict['fine_labels'][40000:])

        test_dict = loadfile(os.path.join(data_path, 'test'))

        X_test = test_dict['data']
        y_test = np.array(test_dict['fine_labels'])

        train_set = (X_train, y_train)
        valid_set = (X_valid, y_valid)
        test_set = (X_test, y_test)

    else:
        raise NotImplementedError('Only implemented the mnist, cifar10 and cifar100 datasets')




    # Load the dataset
    # f = gzip.open(dataset, 'rb')
    # train_set, valid_set, test_set = cPickle.load(f)
    # f.close()


    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
    
    return [train_set, valid_set, test_set]
# end def load_data


def get_rw_path(subdir=''):
    return os.path.join(os.environ['RW_PATH'], subdir)


def get_tmp_path(subdir=''):
    return os.path.join(os.environ['TMP_PATH'], subdir)


def get_share_path(subdir=''):
    return os.path.join(os.environ['SHARE_PATH'], subdir)


def test(*args, **kwargs):
    # just test the usage of args and kwargs
    print args
    print kwargs


if __name__ is "__main__":

    # a = 1
    # b = 2
    # test(a, b, a=a, b=b)
    # test(a, b)

    filename = './test.pkl'

    a = 1
    b = 2
    
    savefile(filename, a, b, a=a, b=b)
    abc = loadfile(filename)
    print abc
    
    savefile(filename, a=a, b=b)
    abc = loadfile(filename)
    print abc
    
    savefile(filename, a, b)
    abc = loadfile(filename)
    print abc

    savefile(filename, (a, b))
    abc = loadfile(filename)
    print abc

    filename = '/mnt/data/datasets/mnist/mnist.pkl.gz'

    datasets = loadfile(filename)

    print datasets[0][0].shape
    print datasets[0][1].shape
    
    
    for data_name in ['mnist', 'cifar10', 'cifar100']:
        datasets = load_data(data_name=data_name)
        print datasets[0][0].shape
        print datasets[0][1].shape
    
    # savefile(filename)
    # abc = loadfile(filename)
    # print abc