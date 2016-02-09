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
course lecturer! 

Settings
===
Network initial settings
```python
initial_setting = {
    "inputs_N"          : 2,       #input dimensions
    "layers"            : [ (20, network.ReLU),(1, network.Sigmoid)], #(number of node, type of nonlinear function)
    "weights_L"         : -0.1, #Random weight initialization bound    
    "weights_H"         : 0.1, #Random weight initialization bound      
    "save"              : False, #haven't implement this one yet  
    "loss"              : network.CrossEntropyLoss #Cost Function you want to use Cross-Entropy or Mean Squared Error 
                   }
```

You can also change some of the settings later(after seeing data), then you need the following code:
```python
replace_value_with_definition("inputs_N",n)
replace_value_with_definition("weights_L",-1.0/n)
replace_value_with_definition("weights_H",1.0/n)
```

Training Curve(Cross-Entropy)
===
![alt tag](https://github.com/wuga214/FullyConnectedDeepNeuralNetwork/blob/master/ANN/src/Training.png)

Data
===
![alt tag](https://github.com/wuga214/FullyConnectedDeepNeuralNetwork/blob/master/DATA.png)
[CIFAR Website](https://www.cs.toronto.edu/~kriz/cifar.html)

Issue
===
1. Haven't implement momentum mechanism yet. So the current code can stuck on saddle points..
2. Haven't implement saving function. So the trained model will lose after finishing the program..
