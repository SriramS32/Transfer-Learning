# Transfer-Learning

Hi!

**Background:** I just started this project two months ago. Based on hearing Deepmind's Demis Hassabis talk about the ability to transfer learning, I wanted to create infrastructure for a network to transfer learning between tasks. Biologically, I thought that transfer learning involves the activation of similar neurons and pathways (or hidden nodes, computationally speaking).

**Repository** contains
- mlpV5.py is main MLP code with transfer learning
- mlpPreTrain.py is MLP code that uses a Deep Belief Network in between transfer learning
- logistic_sgd.py for use by main MLP code & connects last hidden layer to output layer
- getHSF.py processes and pickles image files in grayscale
- resizeHSF.py crops images and resizes to 28*28

Libraries needed: **Theano**, for DBN also need **rbm.py** and **utils.py** from Theano deep learning tutorial

#Code Sketch
The starter MLP python code is from http://deeplearning.net/tutorial/mlp.html.
The transfer learning network first trains on a dataset (HSF_Numbers or HSF_Letters)
- The network is exposed to a sampling of images from the other dataset. Didden nodes can be classified as activated when the node value is above a defined threshold
- The weights to the hidden nodes that were activated above a certain threshold (like a neuron firing through EPSP and IPSP) are preserved and the rest of the weights are re-initialized.
- These weights to activated hidden nodes along with the reinitialized weights are used to newly train the other dataset.

An example of usage:

```javascript
//In command line
python mlpV5.py
//OR
python mlpPreTrain.py
```
For more information, see website.

#Future Plans
Currently, the code examines transfer between numbers and alphabets, but in the future, I hope to explore interactions between significantly different datasets as well.
An algorithm can be developed to designate an appropriate threshold based on the transfer of learning between datasets. The learning rates of weights could be modified rather than preserving the weights and then using a general learning rate. Weights in deeper hidden layers that connect to non-activated H1 nodes could be re-initialized as well.
Need to put in Change flags.

#License
BSD

Sriram Somasundaram

