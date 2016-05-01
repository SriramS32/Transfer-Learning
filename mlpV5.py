"""
The starter MLP code is from http://deeplearning.net/tutorial/mlp.html
@author/editor Sriram Somasundaram


"""
__docformat__ = 'restructedtext en'


import cPickle
import pickle
import os
import sys
import timeit
from PIL import Image

import numpy
from numpy import *


import theano
import theano.tensor as T

from getHSF import getHSF


from logistic_sgd import LogisticRegression, load_data

# HiddenLayer class is responsible for creating weights (initializing if necessary) between inputs and outputs
# MLP class collates hidden layers together (currently 2) along with a log regression layer and defines regularization and errors
# test_mlp() performs the following sequentially:
#     1. Trains the network on one dataset (HSF_Nums or HSF_Letters)
#     2a. Exposes the network to select data from the other dataset
#     2b. Weights from activated hidden nodes (|hnode|>=threshold) are stored in global shared theano variables, whereas the rest are re-initialized
#     3. Trains the network again on the other dataset

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, Whidden=None, Whidden2 = None, Wlog=None):
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

        :type Whidden: theano shared variables
        :param Whidden: first hidden layer

        :type Whidden2: theano shared variable
        :param Whidden: second hidden layer

        """

        #Linking the hidden layers here. If a network is newly initialized (not transfer), then
        #the weights will be set the None and reinitialized.
        #Else (if the network is to be transferred) then use the global
        #theano shared variables of tensor and tensor2

        #Add ons: Can create any number of layers here, just link the inputs and outputs
        if (not transfer):
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh,
                W=Whidden
            )
            self.hiddenLayer2 = HiddenLayer(
                rng =rng,
                input = self.hiddenLayer.output,
                n_in = n_hidden,
                n_out = n_hidden,
                activation = T.tanh,
                W = Whidden2
            )
        else:
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh,
                W=tensor
            )
            self.hiddenLayer2 = HiddenLayer(
                rng = rng,
                input = self.hiddenLayer.output,
                n_in = n_hidden,
                n_out = n_hidden,
                activation = T.tanh,
                W = tensor2
            )
        # The logistic regression layer gets as input the hidden units
        # of the last hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=n_out,
            W=Wlog
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=150,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=100):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
   #Note - transfer is used to check whether test_mlp is running for the first time with new weights or second time with transferred weights
   #Transfer is initialized to be false.
   #a transfer in the if statement will run the code for the Letters data set first and Numbers data set second.
   #(Not transfer) will run the code for the Numbers data set first and Letters data set second. 

   #CHANGE FLAG - edit order datasets are run in and dataset name
    if(transfer):
        #datasets = load_data(dataset)
        f = open('HSFNums.p','rb')
        datasets = pickle.load(f)

    else:
        #datasets = getHSF()
        f = open('HSFLetters2.p','rb')
        datasets = pickle.load(f)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    f.close()
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #total size of valid data is printed
    print 'This is the vector size of the inputs' #
    print train_set_x.get_value(borrow=True).shape #
    print n_train_batches #
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    #Data reduction
    if(transfer):
        train_set_x = train_set_x[0:int(1.0*n_train_batches*batch_size),:]
        train_set_y = train_set_y[0:int(1.0*n_train_batches*batch_size)]


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    #problem is you can't pass weights through here, b/c of gradient descent
    #algorithms use these parameters

    #Numbers have 10 classifications, Letters have 26 classifications.
    #transfer is initialized as false, so depending on which dataset should be run first, edit this
    #CHANGE FLAG - edit the order the network trains in and the number of outputs (n_out)
    if(transfer):
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )
    else:
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=26
        )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #CHANGE FLAG - edit based on the order the network rusn in and the input file name
    inputSize=100 #number of input images sampled from next dataset for transfer calculations
    if(not transfer):
        #f2 = open('HSFLetters2.p','rb')
        #f2 can be changed based on whether letters should be transferred to numbers or v.c.
        f2 = open('HSFNums.p','rb')
        datasetsTransfer = pickle.load(f2)
        train_set_x2, train_set_y2 = datasetsTransfer[0]
        inputs=train_set_x2.get_value(borrow=True) #inputs
        f2.close()
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False


    #opening files to print validation error to
    if(not transfer):
        outFile = open('out.txt','w')
    else:
        outFile = open('outTransfer.txt','w')


    #Inserted code for printing out validation after randomization
    validation_losses = [validate_model(i) for i
                         in xrange(n_valid_batches)]
    this_validation_loss = numpy.mean(validation_losses)
    outFile.write(str(this_validation_loss*100)) #printing the error out to the file, turned to string b/c still using write function
    outFile.write('\n')


    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                outFile.write(str(this_validation_loss*100)) #printing the error out to the file, turned to string b/c still using write function
                outFile.write('\n')
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    #closing file
    outFile.close()
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))



    #Goal of block: Calculate hidden node activations and find which weights to transfer
    #               Create global theano shared variable for the weights to transfer
    if(not transfer):
        
        #Set threshold to determine bounds for activated nodes - Weights leading to activated nodes with absolute values >= threshold
        #will be copied over. Other weights are re-initialized.
        threshold = 0.0
        n_in = 28*28
        #inputs are passed from the train_set_x above
        hidden1W = classifier.hiddenLayer.W.get_value()
        hidden1Wcopy = hidden1W
        #Making a copy of the first hidden layer of weights to be used in calculations for second hidden lyaer of weights
        aveList = []
        #aveList represents the average hidden node activations for layer 1
        print 'starting transfer calculations'
        for i in range(0,n_hidden):
            x = 0
            for j in range(0,inputSize):
                #Design choice to use absolute value b/c a positive activation and a negative activation were both considered important
                x += abs(numpy.tanh(numpy.tensordot(inputs[j,:],hidden1W[:,i],axes=1)))
            aveList.append(x/inputSize)

        print 'ending calculation'

        count = 0
        for i in range(0,n_hidden):
            
            if(aveList[i] < threshold):
                #If the activation is below the threshold, then the weights corresponding leading to that hidden node will be reinitialized
                hidden1W[:,i] = numpy.asarray(
                                rng.uniform(
                                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                                    size=(n_in,1)
                                ),
                                dtype=theano.config.floatX
                            ).flatten()
            else:
                count+=1
        print 'A total number of ' + str(count) + ' H1 nodes passed the threshold'
        
        #saving count of hidden nodes
        outFile3 = open('transfer.txt','w')
        outFile3.write(str(count))
        outFile3.write('\n')



        hidden1Act = numpy.zeros((1,n_hidden))
        #Making a dummy hidden layer variable to edit

        #now for the next hidden layer :)
        hidden2W = classifier.hiddenLayer2.W.get_value()
        aveList = []
        #aveList here represents the average hidden node activations for layer 2
        print 'starting next hidden layer calculation'
        for i in range(0,n_hidden):
            x = 0
            for j in range(0,inputSize):
                for k in range(0,n_hidden):
                    hidden1Act[0][k] = numpy.tanh(numpy.tensordot(inputs[j,:],hidden1Wcopy[:,k],axes=1))
                x += abs(numpy.tanh(numpy.tensordot(hidden1Act[0,:],hidden2W[:,i],axes=1)))
            aveList.append(x/inputSize)
        print 'ending hidden 2 calculation'
        count = 0
        for i in range(0,n_hidden):
            if(aveList[i] < threshold):
                hidden2W[:,i] = numpy.asarray(
                                rng.uniform(
                                    low=-numpy.sqrt(6. / (n_hidden + n_hidden)),
                                    high=numpy.sqrt(6. / (n_hidden + n_hidden)),
                                    size = (n_hidden,1)
                                ),
                                dtype=theano.config.floatX
                            ).flatten()
            else:
                count += 1
        print 'A total number of ' + str(count) + ' H2 nodes passed the threshold'

        outFile3.write(str(count))
        outFile3.close()


        #3 global variables exist. tensor and tensor2 variables are the global theano shared variables for the weights.
        #During the next run, the MLP will be initialized with these weights thereby transferring the weights from this run.
        global transfer
        transfer = True
        global tensor
        global tensor2
        tensor = theano.shared(value=hidden1W,name = 'W', borrow=True)
        tensor2 = theano.shared(value = hidden2W, name = 'tensor2', borrow=True)

        test_mlp() 
    else:
        print 'Thank you for running this transfer program'
        print 'Below are descriptions of files that have been created'
        print 'out.txt         - validation error while training'
        print 'outTransfer.txt - validation error while training after transfer learning'
        print 'transfer.txt    - number of hidden nodes transferred in each layer'


if __name__ == '__main__':
    global transfer
    transfer = False
    #Transfer represents whether the network has been transferred. Initialized as false because network is newly maade here.
    test_mlp()