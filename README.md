# Transfer-Learning

Hi!

**Background:** I just started this project two months ago. Based on hearing Deepmind's Demis Hassabis talk about the ability to transfer learning, I wanted to create infrastructure for a network to transfer learning between tasks. Biologically, I thought that transfer learning involves the activation of similar neurons and pathways (or hidden nodes, computationally speaking).

**Repository** contains
- mlpV3Transfer.py is main MLP code with transfer learning
- mlpV3TransferPictures.py is MLP code that prints out weight maps
- logistic_sgd.py for use by main MLP code & connects last hidden layer to output layer

Libraries needed: **Theano**, **matplotlib**

#Code Sketch
The starter MLP python code is from http://deeplearning.net/tutorial/mlp.html.
The transfer learning code first trains on the MNIST database (http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz).
- A desired set of images from the new database is tested using the trained weight set for activations of hidden nodes
- The weights to the hidden nodes that were activated above a certain threshold (like a neuron firing through EPSP and IPSP) are preserved and the rest of the weights are re-initialized.
mlpV3TransferPictures.py is a slower version of mlpV3Transfer.py as it prints out the weight maps of head hidden node to see exactly what is preserved between data sets.


An example of usage:

```javascript
//In command line
python mlpV3Transfer.py
//OR
python mlpV3TransferPictures.py
```
For more information, see website.

#Future Plans
Currently, the code uses the same database for the initial training and transfer, but in the future, I will format a database of maybe alphabets and see the effects. (Note edit the logistic_sgd.py file which takes in the dataset)
An algorithm can be developed to designate an appropriate threshold based on the transfer of learning between datasets. The learning rates of weights could be modified rather than preserving the weights and then using a general learning rate. Weights in deeper hidden layers that connect to non-activated H1 nodes could be re-initialized as well.

#License
BSD

Sriram Somasundaram


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/SriramS32/transfer-learning/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

