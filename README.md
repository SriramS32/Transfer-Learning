# Transfer-Learning

Hi!

I just started this project two months ago. Based on hearing Deepmind's Demis Hassabis talk about the ability to transfer learning, I wanted to create infrastructure for a network to transfer learning between tasks. Biologically, I thought that transfer learning involves the activation of similar neurons and pathways (or hidden nodes, computationally speaking).

The starter MLP python code is from http://deeplearning.net/tutorial/mlp.html.
The transfer learning code first trains on the MNIST database (http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz). Then, a desired set of images from the new database is tested using the trained weight set for activations of hidden nodes. The weights to the hidden nodes that were activated above a certain threshold (like a neuron firing through EPSP and IPSP) are preserved and the rest of the weights are initialized.

Currently, the code uses the same database for the initial training and transfer, but in the future, I will format a database of maybe alphabets and see the effects.
There is also the option to print out weight maps of each hidden node to see exactly what is preserved between data sets.

Sriram Somasundaram