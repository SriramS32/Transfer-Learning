import cPickle
from PIL import Image
import numpy
import os
import gzip
import theano
import theano.tensor as T

#__docformat__ = 'restructedtext en'

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            #"..",
            #"data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)

    #print train_set[0], train_set[0].shape
    print train_set[0][0,:]
    
    
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
#if __name__ == '__main__':
def changeImage():
    """
    with open('spam.bmp', 'rb') as f:
        data = bytearray(f.read())
    pos = pixel_array_offset + row_size * y + pixel_size * x
    print pos
    data[pos:pos+3] = 255, 255, 255
    with open('eggs.bmp', 'wb') as f:
        f.write(data)
    """
    #     im = Image.open("spam.bmp")
    #print im.format,im.size, im.mode

    #source = im.split()
    #R,G,B = 0,1,2
    #print source[R].point(0)
    #     pixels=im.load()
    """
    a = pixels[1,1]
    b = (0.2989*a[0]/255)+(0.5870*a[1]/255)+(.1140*a[2]/255)
    print b
    """
    names = ()
    bigt = ()
    pathToImageFolder = "/Users/sriramsomasundaram/Desktop/CS/TransferLearning/Python/NewDataset/"
    files_in_dir = os.listdir(pathToImageFolder)
    for file_in_dir in files_in_dir:
        im = Image.open(pathToImageFolder + file_in_dir)
        foo = file_in_dir.split("_", 8)
        names = names + (foo[7],)
        t=()
        pixels = im.load()
        for x in range(28):
            for y in range(28):
                px = pixels[x,y]
                t = t + ((0.2989*px[0]/255)+(0.5870*px[1]/255)+(.1140*px[2]/255),)
        bigt = bigt + (t,)
    print names
"""

    #784 will be the number of inputs per image, 50,000 images, and then labels
    #lets just first put the things into a tuple of 784.
    t = ()
    for x in range(0,784):
        t = t + (x,)
    #... = ('','')  
    print t

    #converting RGB to grayscale
    #R/255, G/255, B/255
    #0.2989*red + 0.5870 +.1140
"""

    #print ''
    #print train_set[1], train_set[1].shape
    #print ''
    #print train_set[2]
    #print ''
    #print train_set[3]
    #foo=str.split("_",8)
    #foo[8]
    #
    #
    #FOR CREATING TUPLE OF NEW DATASET

    #f.close()
changeImage()
#dataset='mnist.pkl.gz'
#load_data(dataset)