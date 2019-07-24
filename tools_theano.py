import numpy as np
import theano
# import theano.tensor as T



def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                             dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                             dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y
# end def shared_dataset


def class_weight_given_y(y):
    '''
    calculate the weights of different classes for class balancing.
    
    class_weight is a vector. len(class_weight) is number of classes in y.
        And sum(class_weight) == number of classes
    
    y is a vector. len(y) is the number of examples. The values of y[i] is from 
        0 to n-1, where n is the number of classes.
        
    Algorithm:
        class_num is the vector with number of examples for each class. 
        len(class_num) = number of classes.
        class_weight[i] = 1 / class_num[i] / sum(1 / class_num) * len(class_num)
    '''
    
    n = int(np.max(y)) + 1  # number of classes
    y = np.array(y)
    
    class_num = np.ones(n)
    for ind in range(n):
        class_num[ind] = np.sum(y == ind)
    
    class_weight = np.ones(n)
    
    den = sum(1. / class_num)
    
    for ind in range(n):
        class_weight[ind] = 1. / class_num[ind] / den * n
    
    return class_weight
# end def class_weight_given_y




# this function can turn any function that only applies on a fixed batch  
# size to function that can take data of any size
# because for theano convnets, the batch_size has to be fixed, so need 
# need to right another 2 functions predict and predict_proba to take 
# arbitrary size input
def batch_to_anysize(batch_size, fcn_hdl, X):
    '''
    X is a np matrix with X.shape[0] = number of examples
    and X.shape[1] = number of feature dimensions (self.img_size ** 2)
    
    fcn_hdl is the function handle that been passed in
    '''        
    n_iter = X.shape[0] / batch_size
    
    num_rem = X.shape[0] % batch_size
    
    # if number of example is not integer times batch_size, need to pad X
    if num_rem > 0:
        n_iter += 1
        X = np.cast[np.float32](np.r_[X, np.zeros((batch_size - num_rem, X.shape[1]))])    
    
    # determine the shape of y by looking at f's output
    y_sample = fcn_hdl(X[:batch_size, :])
    
    if y_sample.ndim == 1:
        y = np.cast[np.float32](np.zeros(X.shape[0]))
    else:
        y = np.cast[np.float32](np.zeros((X.shape[0], y_sample.shape[1]))) 
    
    # for each batch calculate y
    for ind in range(n_iter):
        ind_start = ind * batch_size
        ind_end = (ind + 1) * batch_size
        
        if y.ndim == 1:
            y[ind_start:ind_end] = fcn_hdl(X[ind_start:ind_end, :])
        else:
            y[ind_start:ind_end, :] = fcn_hdl(X[ind_start:ind_end, :])
        
    # if needed cut the padded part of y
    if num_rem > 0:
        if y.ndim == 1:
            y = y[:(n_iter - 1) * batch_size + num_rem]
        else:
            y = y[:(n_iter - 1) * batch_size + num_rem, :]
        
    return y
# end def batch_to_anysize


def cost_batch_to_any_size(batch_size, fcn_hdl, X, y):
    '''
    function for calculating arbitrary input size cost.
    Because of this function takes 2 inputs, which as a different format comparing to other _batch functions, so cannot use batch_to_anysize function.

    and here the last batch has to be discarded as it very hard (if possible) to separate the padded value from the orginal true values

    so to make the predicted cost accurate, use the batch_size that can divide the dataset size

    NOTE: if needed, should make this a generic function
    '''

    assert X.shape[0] == y.shape[0]
    n_batch = X.shape[0] / batch_size
                        
    cost_sum = 0.

    # for each batch calculate cost
    for ind in range(n_batch):
        ind_start = ind * batch_size
        ind_end = (ind + 1) * batch_size
        
        cost_sum += fcn_hdl(X[ind_start:ind_end, :], 
                            y[ind_start:ind_end])

    
    return cost_sum / n_batch
# end def cost_batch_to_any_size
