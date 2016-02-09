# FullyConnectedDeepNeuralNetwork

About this code
===
This is one modularized version of fully connected neural network
In the initial setting, you can specify input dimensions, number of layers and
also number of node in each layers. The activation function is also flexible,
which allows you to change between ReLU and Sigmoid. If you are interested in
using other activation function, please add it in src/network/activations.py

Some implementation of ANN online has similar structure of this one. However, be
aware that the inner implementation are slightly different. My purpose of this
implementation is to show the clear forward and backward structure of Neural
network, but not for high speed running.

This is not GPU version! Do not expect this one can run fast...

Please DO NOT copy this code for any kind of assignment that released by your
college study! 

Training Curve
===
![alt tag](https://github.com/wuga214/FullyConnectedDeepNeuralNetwork/blob/master/ANN/src/Training.png)

Data
===
![alt tag](https://github.com/wuga214/FullyConnectedDeepNeuralNetwork/blob/master/DATA.png)
[CIFAR Website](https://www.cs.toronto.edu/~kriz/cifar.html)
