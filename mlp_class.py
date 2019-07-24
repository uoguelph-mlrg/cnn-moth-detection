import numpy
import theano
import theano.tensor as T
from tools_theano import batch_to_anysize, cost_batch_to_any_size
from fileop import load_data
from layers import HiddenLayer, LogisticRegression, relu

from fit import fit
import logging


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng=numpy.random.RandomState(1234), 
                 n_in=784, n_hidden=500, n_out=10,
                 learning_rate=0.1, n_epochs=200, 
                 batch_size=500, shuffle=True,
                 alpha_l1=0, alpha_l2=0,
                 logreg_activation=T.nnet.softmax,
                 hidden_activation=relu,):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        
        #####################
        # assign parameters #
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.n_out = n_out
        self.shuffle = shuffle
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.rng = rng
        self.hidden_activation = hidden_activation
        self.logreg_activation = logreg_activation
        # assign parameters #
        #####################
        self.build_model()
    # end def __init__    
        
    def build_model(self, flag_preserve_params=False):
        
        logging.info('... building the model')

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels
            

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=self.rng, input=self.x,
                                       n_in=self.n_in, n_out=self.n_hidden,
                                       activation=self.hidden_activation)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=self.n_hidden,
            n_out=self.n_out,
            activation=self.logreg_activation)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
            + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
            + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
        self.cost = self.negative_log_likelihood(self.y) \
            + self.alpha_l1 * self.L1 \
            + self.alpha_l2 * self.L2_sqr
            
            
        self.grads = T.grad(self.cost, self.params)    
        
        # fixed batch size based prediction
        self.predict_proba_batch = theano.function([self.x], 
                                                   self.logRegressionLayer.p_y_given_x)
        self.predict_batch = theano.function([self.x], 
                                             T.argmax(self.logRegressionLayer.p_y_given_x, axis=1))
        self.predict_cost_batch = theano.function([self.x, self.y], self.cost, allow_input_downcast=True)
        
    
    def predict_cost(self, X, y):
        return cost_batch_to_any_size(self.batch_size, self.predict_cost_batch,
                                      X, y)
    # end def predict_cost      
    

    def predict_proba(self, X):
        return batch_to_anysize(self.batch_size, self.predict_proba_batch, X)    
    # end def predict_proba
   
    
    def predict(self, X):
        return batch_to_anysize(self.batch_size, self.predict_batch, X)
    # end def predict
        
        
        
        
        

if __name__ == '__main__':
    from misc import set_quick_logging
    set_quick_logging()
    
    datasets = load_data(data_name='mnist')
    clf = MLP(n_epochs=10, batch_size=200)
    fit(clf, 
        train_set=datasets[0], 
        valid_set=datasets[1],
        test_set=datasets[2], 
        flag_report_test=True,
        flag_report_valid=True,
        early_stop=True)
        
    print clf.predict_proba_batch(datasets[1][0][0:200])
    print clf.predict_batch(datasets[1][0][0:200])
    print clf.predict_proba(datasets[1][0])
    print clf.predict(datasets[1][0])
    print clf.predict_cost_batch(datasets[1][0][0:200], datasets[1][1][0:200])
    print clf.predict_cost(datasets[1][0][0:200], datasets[1][1][0:200])
    print clf.predict_cost(datasets[1][0], datasets[1][1])
