"""
The starter MLP code is from http://deeplearning.net/tutorial/mlp.html
@author/editor Sriram Somasundaram


"""
__docformat__ = 'restructedtext en'


import cPickle
import os
import sys
import timeit
from PIL import Image

import numpy
from numpy import *

"""
#uncomment for creating images
from pylab import *
import matplotlib.cm as cm
import matplotlib.pylab as plt
"""

import theano
import theano.tensor as T

from getHSF import getHSF


from logistic_sgd import LogisticRegression, load_data


# start-snippet-1
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
        # end-snippet-1

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


# start-snippet-2
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

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function

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
        # end-snippet-2 start-snippet-3
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
        # end-snippet-3

        # keep track of model input
        self.input = input


#mlp.LogisticRegression.W
#mlp.HiddenLayer.W
def test_mlp(learning_rate=.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=150,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=100):
#epoch is originally 500, hidden is 500, learning rate is 0.01
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

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
   #Rahul -  a transfer here will run the code for the second data set first. Not transfer, will run the code in the correct order
    if(transfer):
        datasets = load_data(dataset)
    else:
        #datasets = getHSF()
        f = open('HSFBig.p','rb')
        datasets = cPickle.load(f)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    print 'This is the vector size of the inputs' #
    print train_set_x.get_value(borrow=True).shape #
    print n_train_batches #
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

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
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=36
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

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

    # start-snippet-5
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

    #the size is how many in put images should be printed out and how many input images
        #should be averaged on
    inputSize=100
    inputs=train_set_x.get_value(borrow=True) #inputs
    
    """
    print '...printing input images'
    if(not transfer):
        #print out the input images
        fileNameTemplate = 'inputImage'
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        for i in range(0, inputSize):
            a = reshape(inputs[i,:],(28,28))
            matrix = numpy.matrix(a)
            plt.imshow(matrix, interpolation = 'nearest',cmap=plt.cm.Greys)
            savefig(fileNameTemplate+`i`,format = 'png')
    
        #Save a picture of a weight map of the first hidden node before training
        fig = plt.figure() #
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        fileNameTemplate = 'wMapInit'
        for i in range(0,n_hidden):
            a = classifier.hiddenLayer.W.get_value()[:,i]
            a = reshape(a,(28,28))
            matrix = numpy.matrix(a)
            plt.imshow(matrix,interpolation='nearest',cmap=plt.cm.Greys)
            #ocean
            savefig(fileNameTemplate+`i`,format='png') #
    
    """
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











    if(not transfer):
        
        #Save a picture of a weight map of the first hidden node after training
        """
        print '...printing weight maps of H1 (hidden nodes in layer 1) after training'
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        fileNameTemplate = 'wMapTrained'
        for i in range(0,n_hidden):
            a = classifier.hiddenLayer.W.get_value()[:,i]
            a = reshape(a,(28,28))
            matrix = numpy.matrix(a)
            plt.imshow(matrix,interpolation='nearest',cmap=plt.cm.Greys)
            #ocean
            savefig(fileNameTemplate+`i`,format='png')
        """
        
        #Copy over weights that lead to activated nodes
        threshold = 0.6
        n_in = 28*28
        #inputs as d are passed from the train_set_x above
        hidden1W = classifier.hiddenLayer.W.get_value()
        #just being safe for now with all the copies, can change this later
        aveList = []
        print 'starting transfer calculations'
        for i in range(0,n_hidden):
            x = 0
            for j in range(0,inputSize):
                x += abs(numpy.tanh(numpy.tensordot(inputs[j,:],hidden1W[:,i],axes=1)))
            aveList.append(x/inputSize)

        print 'ending calculation'

        count = 0
        for i in range(0,n_hidden):
            #print 'hidden'
            #print i

            #if(T.dot(input[0,:],classifier.hiddenLayer.W.get_value()[:,i])<threshold):
            #problem initially had [0,0] to select the value out of a possible matrix, was not
            #a problem and in fact wouldn't work like that, so I had to remove that part
            #Another problem was for some reason column of new matrix was treated as 784 1D
            #unlike in the file tensorPractice.py, when I created the matrix myself, so
            #I had to use the flatten function to flatten the array which was 784 by 1 to 784
            #because values in aveList are added absolute values, this if statement is enough. EDIT from mlpGlob.py, also
            #the print statements are changed
            if(aveList[i] < threshold):
                #randomize it
                #print 'did not pass the threshold'
                hidden1W[:,i] = numpy.asarray(
                                rng.uniform(
                                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                                    size=(n_in,1)
                                ),
                                dtype=theano.config.floatX
                            ).flatten()
            else:
                #print 'passed the threshold'
                count+=1
            #print 'with a value of'
            #print aveList[i]
            #print ''
        print 'A total number of ' + str(count) + ' H1 nodes passed the threshold'
        
        #printing out count of hidden nodes
        outFile3 = open('transfer.txt','w')
        outFile3.write(str(count))
        outFile3.write('\n')






        #now for the next hidden layer :)
        hidden2W = classifier.hiddenLayer2.W.get_value()
        aveList = []
        print 'starting next hidden layer calculation'
        for i in range(0,n_hidden):
            x = 0
            for j in range(0,n_hidden):
                x += abs(numpy.tanh(numpy.tensordot(inputs[j,:],hidden2W[:,i],axes=1)))
                aveList.append(x/inputSize)
        print 'ending hidden 2 calculation'
        count = 0
        for i in range(0,n_hidden):
            if(aveList[i] < threshold):
                hidden2W[:,i] = numpy.asarray(
                                rng.uniform(
                                    low=-numpy.sqrt(6. / (n_hidden + n_hidden)),
                                    high=numpy.sqrt(6. / (n_hidden + n_hidden)),
                                    size = (n_in,1)
                                ),
                                dtype=theano.config.floatX
                            ).flatten()
            else:
                count += 1
        print 'A total number of ' + str(count) + ' H2 nodes passed the threshold'

        outFile3.write(str(count))









        """
        #print the weight maps (some will be randomized to pass on the next iteration of training, others will be preserved)
        print '...printing H1 weight maps after transfering learning'
        fileNameTemplate = 'wMapRelevant'
        for i in range(0,n_hidden):
            a = hidden1W[:,i]
            a = reshape(a,(28,28))
            matrix = numpy.matrix(a)
            plt.imshow(matrix,interpolation='nearest',cmap=plt.cm.Greys)
            #ocean
            savefig(fileNameTemplate+`i`,format='png')
        """

        global transfer
        transfer = True
        global tensor
        global tensor2
        tensor = theano.shared(value=hidden1W,name = 'W', borrow=True)
        #tensor = theano.shared(value = classifier.hiddenLayer.W.get_value(), name = 'tensor', borrow=True)
        tensor2 = theano.shared(value = classifier.hiddenLayer2.W.get_value(), name = 'tensor2', borrow=True)
        #tensor2 = None

        test_mlp()
    else:
        print 'Thank you for running this transfer program'
        print 'Below are descriptions of files that have been created'
        print 'out.txt         - validation error while training'
        print 'outTransfer.txt - validation error while traning after transfer learning'
        """
        print ''
        print 'wMapInit group    -  weight maps after initialization'
        print 'wMapTrained group -  weight maps after training'
        print 'wMapTransfer group    -  weight maps after transfering'
        print '     relevant weight maps to activated nodes are identical to wMapTrained'
        print '     non relevant weight maps are re-initialized'
        """


if __name__ == '__main__':
    global transfer
    transfer = False
    test_mlp()