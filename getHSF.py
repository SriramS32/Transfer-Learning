from PIL import Image
import numpy
import os 
import theano 
import theano.tensor as T
import cPickle

#Converting a-z & A-Z -> 0-25
#Converting any integer to its corresponding int64 value.
def convert(str):
    val = ord(str.lower())
    if(val>= 97 and val<= 122):
        return numpy.int64(val-97)
    else:
        return numpy.int64(str)

"""
Files in a directory are iterated through and opened using Python Image library (PIL).
The labels of the images are extracted and put into names tuple. The pixel values of the image
are placed into t1 in grayscale (converted from RGB). The (1, 784) (from the 28*28 pixel size) 
tuple is put into a numpy array and stacked onto bigt (type float32). Then, the names are
converted to int64 type using convert() and placed into parseNames. The
data are concatenated with the labels and split into train, test, and validation (dev).  
The shared dataset function is used to load dataset into shared variables and allow 
Theano to use GPU memory. Finally, the datasets are concatenated together and pickled.

"""
def getHSF():
    train_dataset, dev_dataset, test_dataset = (), (), ()
    names = ()
    parseNames = [] 
    bigt = []

    #pathToImageFolder = "/Users/sriramsomasundaram/Desktop/CS/TransferLearning/MergedResized/"
    #pathToImageFolder = "/Users/sriramsomasundaram/Desktop/HSFresized/"
    #make sure you have a / at the end of the file name
    pathToImageFolder = "/home/rcf-proj/sn/sriramso/HSFresizedtest/"
    files_in_dir = os.listdir(pathToImageFolder)
    bigt = numpy.zeros((0,784),dtype='float32')
    count = 0
    for file_in_dir in files_in_dir:
        try:        
            im = Image.open(pathToImageFolder + file_in_dir)
        except: 
            continue
        foo = file_in_dir.split("_", 8)
        #Retreiving image labels
        names = names + (foo[7],)
        t1 = numpy.zeros((28,28),dtype = 'float32')
        pixels = im.load()
        for x in range(28):
            for y in range(28):
                px = pixels[x,y]
                #Converting to grayscale
                t1[y,x] = (0.2989*px[0]/255)+(0.5870*px[1]/255)+(.1140*px[2]/255)
        t = numpy.reshape(t1,(1,784))
        bigt = numpy.vstack((bigt,t))
        #Reshaping image and adding to bigger tuple
        if count % 1000 == 0:
            print count
        count+=1
    bigt = numpy.array(bigt,dtype='float32')
    for x in names:
        parseNames.append(convert(x))
        #converting name labels to integers
    parseNames = numpy.array(parseNames,dtype='int64')

    # breaking dataset into train test and dev set (80%, 10%, 10% split)   
    num_datapoints = bigt.shape[0]
    train_bigt, train_parseNames = bigt[0:int(.8*num_datapoints),:], parseNames[0:int(.8*num_datapoints)]
    dev_bigt, dev_parseNames = bigt[int(.8*num_datapoints):int(.9*num_datapoints),:], parseNames[int(.8*num_datapoints):int(.9*num_datapoints)]
    test_bigt, test_parseNames = bigt[int(.9*num_datapoints):,:], parseNames[int(.9*num_datapoints):]


    train_dataset = train_dataset + (train_bigt,)
    train_dataset = train_dataset + (train_parseNames,)
    dev_dataset = dev_dataset + (dev_bigt,)
    dev_dataset = dev_dataset + (dev_parseNames,)
    test_dataset = test_dataset + (test_bigt,)
    test_dataset = test_dataset + (test_parseNames,)
   
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_dataset)
    valid_set_x, valid_set_y = shared_dataset(dev_dataset)
    train_set_x, train_set_y = shared_dataset(train_dataset)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    #return rval
    cPickle.dump(rval,open('HSFLetters2.p','wb'))


if __name__ == "__main__":
   getHSF()
