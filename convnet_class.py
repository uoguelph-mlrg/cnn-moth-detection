"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import copy
import numpy as np
import theano
import theano.tensor as T
from tools_theano import batch_to_anysize, cost_batch_to_any_size
from fileop import load_data
from layers import HiddenLayer, LogisticRegression, LeNetConvPoolLayer, relu
from fit import fit
import logging
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class lenet5(object):
    
    def __init__(self, learning_rate=0.1, n_epochs=200, nkerns=[20, 50], 
                 batch_size=500, img_size=28, img_dim=1, filtersize=(5, 5), 
                 poolsize=(2, 2), num_hidden=500, num_class=10, shuffle=True, 
                 cost_type ='nll_softmax', 
                 alpha_l1 = 0, alpha_l2 = 0, alpha_entropy=0,
                 rng = np.random.RandomState(23455),
                 logreg_activation=T.nnet.softmax,
                 hidden_activation=relu,
                 conv_activation=relu):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        
        #####################
        # assign parameters #
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dim = img_dim
        self.filtersize = filtersize
        self.poolsize = poolsize
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.shuffle = shuffle
        self.cost_type = cost_type
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.alpha_entropy = alpha_entropy
        self.rng = rng
        self.logreg_activation = logreg_activation
        self.conv_activation = conv_activation
        self.hidden_activation = hidden_activation
        # assign parameters #
        #####################
        
        # call build model to build theano and other expressions
        self.build_model()
        self.build_functions()
    # end def __init__     


    def build_model(self, flag_preserve_params=False):
    
        
        ###################
        # build the model #
        logging.info('... building the model')
        
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        
        # self.y = T.ivector('y')  
        # the labels are presented as 1D vector of
        # [int] labels, used to represent labels given by 
        # data
        
        # the y as features, used for taking in intermediate layer "y" values                    
        self.y = T.matrix('y')   
        

        
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        self.layer0_input = self.x.reshape((self.batch_size, self.img_dim, self.img_size, self.img_size))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        self.layer0 = LeNetConvPoolLayer(self.rng, input=self.layer0_input,
                                         image_shape=(self.batch_size, self.img_dim, self.img_size, self.img_size),
                                         filter_shape=(self.nkerns[0], self.img_dim, 
                                                       self.filtersize[0], self.filtersize[0]),
                                         poolsize=(self.poolsize[0], self.poolsize[0]),
                                         activation=self.conv_activation)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        
        self.img_size1 = (self.img_size - self.filtersize[0] + 1) / self.poolsize[0]
        
        self.layer1 = LeNetConvPoolLayer(self.rng, input=self.layer0.output,
                                         image_shape=(self.batch_size, self.nkerns[0], 
                                                      self.img_size1, self.img_size1),
                                         filter_shape=(self.nkerns[1], self.nkerns[0], 
                                                       self.filtersize[1], self.filtersize[1]), 
                                         poolsize=(self.poolsize[1], self.poolsize[1]),
                                         activation=self.conv_activation)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        self.layer2_input = self.layer1.output.flatten(2)
        
        self.img_size2 = (self.img_size1 - self.filtersize[1] + 1) / self.poolsize[1]
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(self.rng, input=self.layer2_input, 
                                  n_in=self.nkerns[1] * self.img_size2 * self.img_size2,
                                  n_out=self.num_hidden, 
                                  activation=self.hidden_activation)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, 
                                         n_in=self.num_hidden, 
                                         n_out=self.num_class,
                                         activation=self.logreg_activation)
        
        
        # regularization term
        self.decay_hidden = self.alpha_l1 * abs(self.layer2.W).sum() + \
            self.alpha_l2 * (self.layer2.W ** 2).sum()
            
        self.decay_softmax = self.alpha_l1 * abs(self.layer3.W).sum() + \
            self.alpha_l2 * (self.layer3.W ** 2).sum()
        
        
        # there's different choices of cost models
        if self.cost_type == 'nll_softmax':
            # the cost we minimize during training is the NLL of the model
            self.y = T.ivector('y')  # index involved so has to use integer
            self.cost = self.layer3.negative_log_likelihood(self.y) + \
                self.decay_hidden + self.decay_softmax + \
                self.alpha_entropy * self.layer3.p_y_entropy
                
                
        elif self.cost_type == 'ssd_softmax':
            self.cost = T.mean((self.layer3.p_y_given_x - self.y) ** 2) + \
                self.decay_hidden + self.decay_softmax
            
        elif self.cost_type == 'ssd_hidden':
            self.cost = T.mean((self.layer2.output - self.y) ** 2) + \
                self.decay_hidden
        
        elif self.cost_type == 'ssd_conv':
            self.cost = T.mean((self.layer2_input - self.y) ** 2)
        
        # create a list of all model parameters to be fit by gradient descent
        
        # preserve parameters if the exist, used for keep parameter while 
        # changing
        # some of the theano functions
        # but the user need to be aware that if the parameters should be kept 
        # only if the network structure doesn't change
        
        if flag_preserve_params and hasattr(self, 'params'):
            pass
            params_temp = copy.deepcopy(self.params)
        else:
            params_temp = None
        
        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
            
        # if needed, assign old parameters
        if flag_preserve_params and (params_temp is not None):
            for ind in range(len(params_temp)):
                self.params[ind].set_value(params_temp[ind].get_value(), borrow=True)


        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params, disconnected_inputs='warn')
        
        # error function from the last layer logistic regression
        self.errors = self.layer3.errors 
        # the above line will cause the crash of cPickle, need to use 
        # __getstate__ and __setstate__ to deal with it
        
        # build the model #
        ###################

    # end def build_model       

    def build_functions(self):
        # prediction methods
        
        self.fcns = {}

        self.fcns['predict_proba_batch'] = theano.function([self.x], self.layer3.p_y_given_x)
        self.fcns['predict_batch'] = theano.function([self.x], T.argmax(self.layer3.p_y_given_x, axis=1))
        self.fcns['predict_hidden_batch'] = theano.function([self.x], self.layer2.output)
        self.fcns['predict_convout_batch'] = theano.function([self.x], self.layer2_input)
        # self.predict_proba_batch = theano.function([self.x], self.layer3.p_y_given_x)
        # self.predict_batch = theano.function([self.x], T.argmax(self.layer3.p_y_given_x, axis=1))
        # self.predict_hidden_batch = theano.function([self.x], self.layer2.output)
        # self.predict_convout_batch = theano.function([self.x], self.layer2_input)
        
        # cost function for a single batch
        # suitable for negative_log_likelihood input y
        self.fcns['predict_cost_batch'] = theano.function([self.x, self.y], self.cost, allow_input_downcast=True)
        
        # predict entropy
        # this function is for debugging purpose
        self.fcns['predict_entropy_batch'] = theano.function([self.x], self.layer3.p_y_entropy)
        
    
    
    def predict_cost(self, X, y):
        return cost_batch_to_any_size(self.batch_size, self.fcns['predict_cost_batch'], X, y)
    # end def predict_cost  


    def predict_proba(self, X):
        return batch_to_anysize(self.batch_size, self.fcns['predict_proba_batch'], X)
    # end def predict_proba
   
    
    def predict(self, X):
        return batch_to_anysize(self.batch_size, self.fcns['predict_batch'], X)
    # end def predict
    
    
    def predict_hidden(self, X):
        return batch_to_anysize(self.batch_size, self.fcns['predict_hidden_batch'], X)
    # end def predict_hidden
    
    
    def predict_convout(self, X):
        return batch_to_anysize(self.batch_size, self.fcns['predict_convout_batch'], X)
    # end def predict_convout
       
    
    # copy weight parameters from another lenet5
    def copy_weights(self, clf):
        
        # check the whether should copy
        if type(clf) is lenet5 and self.nkerns == clf.nkerns and self.img_size == clf.img_size and self.filtersize == clf.filtersize and self.poolsize == clf.poolsize and self.num_hidden == self.num_hidden and self.num_class == clf.num_class:
            self.set_weights(clf.params)
        else:
            print "Weight's not copied, the input classifier doesn't match the original classifier"   
    # end def copy_params
    
    
    def set_weights(self, params_other):
        '''
        set weights from other trained network or recorded early stopping.
        Use this function with caution, because it doesn't check whether the 
        weights are safe to copied
        '''
        for ind in range(len(params_other)):
            self.params[ind].set_value(params_other[ind].get_value(), borrow=True)   
    # end def set_weights    
    
        
    #################################
    # dealing with cPickle problems #
    
    def __getstate__(self):
        print '__getstate__ executed'


        saved_weights = []

        for param in self.params:
            saved_weights.append(param.get_value())

        list_to_del = ["index", "x", "y", "layer0_input",
                       "layer0", "img_size1", "layer1", "layer2_input",
                       "img_size2", "layer2", "layer3", "decay_hidden",
                       "decay_softmax", "cost",
                       "params", "grads",
                       "errors", "fcns", ]

        state = self.__dict__.copy()

        state['saved_weights'] = saved_weights
        for key in state.keys():
            if key in list_to_del:
                del state[key]
        # del state['errors']
        # del state['fcns']
        return state
    # end def __getstate__
        
        
    def __setstate__(self, state):
        print '__setstate__ executed'
        self.__dict__ = state
        # self.errors = self.layer3.errors
         
        self.build_model()
        self.build_functions()

        for ind in range(len(state['saved_weights'])):
            self.params[ind].set_value(state['saved_weights'][ind])
    # end def __setstate__
    
    # dealing with cPickle problems #
    #################################

# end class lenet5


if __name__ == '__main__':
    from misc import set_quick_logging
    set_quick_logging()

    from fileop import savefile

    clf = lenet5(n_epochs=3, nkerns=[32, 64], batch_size=256)

    savefile('/tmp/clf.pkl', clf)

    batch_size = 256

    datasets = load_data(data_name='mnist')
    
    clf1 = lenet5(n_epochs=3, nkerns=[32, 64], batch_size=256)
    fit(clf1, 
        train_set=datasets[0], 
        valid_set=datasets[1],
        test_set=datasets[2], 
        flag_report_test=True,
        flag_report_valid=True,
        early_stop=True)

    import time


    time_bgn = time.clock()
    for i in range(10):
        clf.predict_proba(datasets[1][0])

    time_end = time.clock()
    print time_end - time_bgn

    time_bgn = time.clock()
    for i in range(10):
        clf1.predict_proba(datasets[1][0])

    time_end = time.clock()
    print time_end - time_bgn


    print clf.predict_proba(datasets[1][0])
    print clf.predict(datasets[1][0])
    print clf.predict_cost(datasets[1][0][0:batch_size], datasets[1][1][0:batch_size])
    print clf.predict_cost(datasets[1][0], datasets[1][1])
