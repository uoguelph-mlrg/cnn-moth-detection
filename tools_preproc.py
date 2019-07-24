'''
Tools for prerocessing training / test data
'''

from pylearn2.expr.preprocessing import global_contrast_normalize
# from pylearn2.datasets.preprocessing import lecun_lcn
import numpy as np
from theano import function, tensor
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.datasets.preprocessing import gaussian_filter
import copy


def centersphere(X, transpose=True, sphere=True, center=True, method='PCA', dim=None, A=None):
    """
    The function  Y = centersphere(X,s,c,method,dim) centers and spheres the
    data and possibly applies PCA.

    Converted to Python from Geoff Hinton's MATLAB code.

    :param X: : data-matrix
    :param transpose: set to True if X is of shape [# examples, # dimensions]
    :param s: True if we want to sphere the data
    :param c: True if we want to center the data.
    :param method: one of 'PCA' or 'ZCA'
    :rval Y: whitened data
    """
    # original file downloaded from 
    # http://hg.assembla.com/LeDeepNet/file/tip/utils/whiten.py
    # 
    # When A=None, will fit A, when A is given, will directly use A
    # 

    X = X.T if transpose else X
    [D, N] = X.shape

    # here the original flag variable was deleted as I don't need to determine which dimension is feature and which is data

    Z = X


    if center:
        # expand_dims is for broadcasting the matrix
        Z = Z - np.expand_dims(np.mean(Z, axis=1), 1)

    if sphere:
        # NOTE according to Andrew Ng's lecture about svd implementaiton of PCA, it should be [U,L,V] = linalg.svd(Z), but simply doing this here gives wrong results. 
        # Probably need also to change the code inside "if method=='ZCA'"
        # Here it tries to do of covariance, while Andrew Ng claims that instead we can do svd directly on Z, there should be some equivalence here.
        # This can be done later 

        if A is None:
            C = np.dot(Z, Z.T) / N
            [U, L, V] = linalg.svd(C)

            # reduce dimensionality to dim
            if dim:
                U = U[:, :dim]
                L = L[:dim] 
                   
            L = np.diag(L[:])
            if method == 'ZCA':
                A = np.dot(U, np.dot(np.sqrt(np.linalg.inv(L)), U.T))
            else:
                A = np.dot(np.sqrt(np.linalg.inv(L)), U.T)


        
        Z = np.dot(A, Z)

    Y = Z

    
    return Y.T if transpose else Y, A
# end def centersphere



class preprocessor(object):
    """
    everything is set to false by default

    parameters:
        ZCA: 
            zca_mat

        lcn:
            see preproc LeCunLCN
        
        global_contrast_normalize, right now all default value
            X, input matrix
            scale=1., 
            subtract_mean=True, 
            use_std=False,
            sqrt_bias=0., 
            min_divisor=1e-8, 

        flag_gcn is defaultly set to false as the ZCA results after gcn seems problematic
    """
    def __init__(self, flag_gcn=False, flag_zca=False, flag_lcn=False, img_shape=(28, 28, 3), kernel_size=5):
        
        self.zca_mat = None
        self.flag_gcn = flag_gcn
        self.flag_zca = flag_zca
        self.flag_lcn = flag_lcn
        self.img_shape = img_shape  # only useful when flag_lcn=True
        self.kernel_size = kernel_size  # only useful when flag_lcn=True
        
        if flag_lcn:
            self.lcn = LeCunLCN()
    # end def __init__


    def fit(self, X):
        
        if self.flag_gcn:
            X = global_contrast_normalize(X)

        if self.flag_lcn:
            X = self.lcn_transform(X)

        if self.flag_zca:
            X, self.zca_mat = centersphere(X, method='ZCA')
      
        return X
    # end def fit

    def transform(self, X):
        
        if self.flag_gcn:
            X = global_contrast_normalize(X)

        if self.flag_lcn:
            X = self.lcn_transform(X)

        if self.flag_zca:
            X, _ = centersphere(X, method='ZCA', A=self.zca_mat)

        return X
    # end def transform


    def lcn_transform(self, X):
        # NOTE: lcn happens in place, if don't want to modify the original X, use copy.deepcopy(X) as input instead of X
        # X.shape should be (num_img, channel * row * col)
        # self.img_shape is (row, col, channel)
        max_size = 50000

        # reshape X to do lcn
        if X.shape[1] == np.prod(self.img_shape):
            # color image case
            X = np.rollaxis(X.reshape(X.shape[0], self.img_shape[2], 
                                      self.img_shape[0], 
                                      self.img_shape[1]), 
                            1, 4)

        elif X.shape[1] == np.prod(self.img_shape[:2]):
            # grey image case
            X = X.reshape(X.shape[0], 
                          self.img_shape[0], self.img_shape[1])

        else:
            ValueError('X.shape does not match img_shape!')


        X = self.lcn.transform(X, self.kernel_size, max_size=max_size)

        # reshape X back
        if X.ndim == 4:
            # color image case
            X = np.rollaxis(X, 3, 1).reshape(X.shape[0], np.prod(X.shape[1:]))

        elif X.ndim == 3 or (X.ndim == 4 and X.shape[3] == 1):
            # grey image case
            X = X.reshape(X.shape[0], np.prod(X.shape[1:]))

        return X
    # end def lcn_transform

# end class preprocessor


def gen_fcn(batch_size, img_shape, kernel_size, data_type='float32', threshold=1e-4):
    '''
    generate theano function for doing lecun lcn of a given setting
    modified from lecun_lcn in pylearn2.datasets.preprocessing 

    currently data_type can only be float32
    if not, will report error saying input and kernel should be the same type
    and kernel type is float32
    '''

    X = tensor.matrix(dtype=data_type)
    X = X.reshape((batch_size, img_shape[0], img_shape[1], 1))

    filter_shape = (1, 1, kernel_size, kernel_size)
    filters = sharedX(gaussian_filter(kernel_size).reshape(filter_shape))

    input_space = Conv2DSpace(shape=img_shape, num_channels=1)
    transformer = Conv2D(filters=filters, batch_size=batch_size,
                         input_space=input_space,
                         border_mode='full')
    convout = transformer.lmul(X)

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_size / 2.))
    centered_X = X - convout[:, mid:-mid, mid:-mid, :]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    transformer = Conv2D(filters=filters,
                         batch_size=batch_size,
                         input_space=input_space,
                         border_mode='full')
    sum_sqr_XX = transformer.lmul(X ** 2)

    denom = tensor.sqrt(sum_sqr_XX[:, mid:-mid, mid:-mid, :])
    per_img_mean = denom.mean(axis=[1, 2])
    divisor = tensor.largest(per_img_mean.dimshuffle(0, 'x', 'x', 1), denom)
    divisor = tensor.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = tensor.flatten(new_X, outdim=3)

    f = function([X], new_X)
    return f


class LeCunLCN(object):
    """
    Yann LeCun's local contrast normalization
    adopted from pylearn2:
        function lecun_lcn and class LeCunLCN from module pylearn2.datasets.preprocessing 
    """

    def __init__(self):

        self.fcn_dict = {}  # dictionary storing compiled theano functions


    def transform_1c(self, X, kernel_size):
        '''
        for transform single channel
        X is the input matrix, X.shape should be (batch_size, rows, cols) or (batch_size, rows, cols, 1)
        '''
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        img_shape = X.shape[1:3]
        batch_size = X.shape[0]

        params_tuple = (batch_size, img_shape, kernel_size)

        if params_tuple not in self.fcn_dict:
            self.fcn_dict[params_tuple] = gen_fcn(*params_tuple)

        return self.fcn_dict[params_tuple](X)


    def transform(self, X, kernel_size, flag_inplace=False, max_size=None):
        '''
        X is the input matrix, X.shape should be (batch_size, rows, cols, channel)

        NOTE: lcn happens in place, if need to keep the original data, make a copy outside of this function. This can be done by simply doing transform(copy.deepcopy(X))

        flag_inplace is not used for now
        max_size is the maximum number that a single transform can deal with
        usually set as None
        if set recommend using 50000 (number of cifar10 images)

        '''

        # if number of images larger than max_size, need to 
        if max_size is not None and X.shape[0] > max_size:
            print "batch_size larger than max_size, divide X into small pieces"
            num_rounds = int(np.ceil(X.shape[0] / float(max_size)))
            bgn_list = map(lambda x: x * max_size, range(num_rounds))
            end_list = map(lambda x: x, bgn_list[1:]) + [X.shape[0]]

            for ind_bgn, ind_end in zip(bgn_list, end_list):
                assert X[ind_bgn:ind_end].shape[0] <= max_size
                X[ind_bgn:ind_end] = self.transform(X[ind_bgn:ind_end], kernel_size=kernel_size, flag_inplace=flag_inplace, max_size=max_size)

            return X



        if X.ndim == 3 or (X.ndim == 4 and X.shape[3] == 1):
            X[:] = self.transform_1c(X, kernel_size=kernel_size)
            return X
        elif X.ndim == 4 and X.shape[3] > 1:
            for channel in range(X.shape[3]):
                X[:, :, :, channel] = self.transform_1c(X[:, :, :, channel], kernel_size=kernel_size)
            return X

        else:
            ValueError('X.shape should be (batch_size, rows, cols, channel)')


def lcn_2d(im, sigmas=[1.591, 1.591]):
    """ Apply local contrast normalization to a square image.
    Uses a scheme described in Pinto et al (2008)
    Based on matlab code by Koray Kavukcuoglu
    http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz

    data is 2-d
    sigmas is a 2-d vector of standard devs (to define local smoothing kernel)

    Example
    =======
    im_p = lcn_2d(im,[1.591, 1.591])
    """

    # assert(issubclass(im.dtype.type, np.floating))
    im = np.cast[np.float](im)

    # 1. subtract the mean and divide by std dev
    mn = np.mean(im)
    sd = np.std(im, ddof=1)

    im -= mn
    im /= sd

    # # 2. compute local mean and std
    # kerstring = '''0.0001    0.0005    0.0012    0.0022    0.0027    0.0022    0.0012    0.0005    0.0001
    #     0.0005    0.0018    0.0049    0.0088    0.0107    0.0088    0.0049    0.0018    0.0005
    #     0.0012    0.0049    0.0131    0.0236    0.0288    0.0236    0.0131    0.0049    0.0012
    #     0.0022    0.0088    0.0236    0.0427    0.0520    0.0427    0.0236    0.0088    0.0022
    #     0.0027    0.0107    0.0288    0.0520    0.0634    0.0520    0.0288    0.0107    0.0027
    #     0.0022    0.0088    0.0236    0.0427    0.0520    0.0427    0.0236    0.0088    0.0022
    #     0.0012    0.0049    0.0131    0.0236    0.0288    0.0236    0.0131    0.0049    0.0012
    #     0.0005    0.0018    0.0049    0.0088    0.0107    0.0088    0.0049    0.0018    0.0005
    #     0.0001    0.0005    0.0012    0.0022    0.0027    0.0022    0.0012    0.0005    0.0001'''
    # ker = []
    # for l in kerstring.split('\n'):
    #     ker.append(np.fromstring(l, dtype=np.float, sep=' '))
    # ker = np.asarray(ker)

    # lmn = scipy.signal.correlate2d(im, ker, mode='same', boundary='symm')
    # lmnsq = scipy.signal.correlate2d(im ** 2, ker, mode='same', boundary='symm')

    lmn = gaussian_filter(im, sigmas, mode='reflect')
    lmnsq = gaussian_filter(im ** 2, sigmas, mode='reflect')

    lvar = lmnsq - lmn ** 2
    # lvar = np.where( lvar < 0, lvar, 0)
    np.clip(lvar, 0, np.inf, lvar)  # items < 0 set to 0
    lstd = np.sqrt(lvar)

    np.clip(lstd, 1, np.inf, lstd)

    im -= lmn
    im /= lstd

    return im
# end def lcn_2d


if __name__ == '__main__':
    from pylearn2.datasets import cifar10
    import matplotlib.pylab as plt
    from classification import load_initial_data
    from fileop import loadfile
    import copy

    flag_cifar10 = False
    flag_covmat = False

    if flag_cifar10:
        img_shape = (32, 32, 3)
        train = cifar10.CIFAR10(which_set="train", one_hot=True)
        test = cifar10.CIFAR10(which_set="test", one_hot=True)
        X = train.X
        X_test = test.X
    
    else:
        # use moth data for test
        img_shape = (28, 28, 3)
        config = loadfile('config.yaml')
        X, _, X_test, _ = \
            load_initial_data(data_path=config['data_path'],
                              target_width=config['target_width'], 
                              target_height=config['target_height'], 
                              flag_rescale=config['flag_rescale'],
                              flag_multiscale=config['flag_multiscale'],
                              detect_width_list=config['detect_width_list'], 
                              detect_height_list=config['detect_height_list'],
                              flag_take_valid=config['flag_take_valid'],
                              flag_rgb=config['flag_rgb'],
                              flag_rot_aug_train=config['flag_rot_aug'],
                              flag_rot_aug_valid=config['flag_rot_aug'],
                              ) 

    

    # X = X[:5000]
    # X_test = X_test[:5000]


    scaler = preprocessor(flag_gcn=False, flag_zca=False, flag_lcn=True)

    # X_proc, A = centersphere(X, method='ZCA')
    # X_test_proc, _ = centersphere(X_test, method='ZCA', A=A)

    X_proc = scaler.fit(copy.deepcopy(X))
    X_test_proc = scaler.transform(copy.deepcopy(X_test))


    if flag_covmat:
        covmat_zca = np.cov(X_proc.T)
        covmat_test_zca = np.cov(X_test_proc.T)
        
        plt.figure()
        plt.imshow(covmat_zca, vmin=0., vmax=1.)
        plt.show()

        plt.figure()
        plt.imshow(covmat_test_zca, vmin=0., vmax=1.)
        plt.show()


    def rot_data(data, img_shape=img_shape):
        return np.rollaxis(data.reshape(data.shape[0], img_shape[2], img_shape[0], img_shape[1]), 1, 4)
    

    def show_img(data, ind_img):
        img = data[ind_img, :, :, :]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        plt.imshow(img)


    X = rot_data(X)
    X_proc = rot_data(X_proc)
    X_test = rot_data(X_test)
    X_test_proc = rot_data(X_test_proc)

    for ind_img in [1, 101, 1001, 10001]:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title('train original')
        show_img(X, ind_img)
        plt.subplot(2, 2, 2)
        plt.title('train processed')
        show_img(X_proc, ind_img)
        plt.subplot(2, 2, 3)
        plt.title('test original')
        show_img(X_test, ind_img)
        plt.subplot(2, 2, 4)
        plt.title('test processed')
        show_img(X_test_proc, ind_img)
        plt.show()

