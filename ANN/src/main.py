'''
Created on Feb 6, 2016

@author: Wuga
'''
#===============================================================================
# This is one modularized version of fully connected neural network
# In the initial setting, you can specify input dimensions, number of layers and
# also number of node in each layers. The activation function is also flexible,
# which allows you to change between ReLU and Sigmoid. If you are interested in
# using other activation function, please add it in src/network/activations.py
# 
# Some implementations of ANN OL have similar structure of this one. However, be
# aware that the inner implementation are slightly different. My purpose of this
# implementation is to show the clear forward and backward structure of Neural
# network, but not for high speed running.
#
# This is not GPU version! Do not expect this one can run fast...
#
# Please DO NOT copy this code for any kind of assignment that released by your
# college study! 
#
# Wuga
# PhD student @ Oregon State University
#===============================================================================

import network
import data
import visualization

initial_setting = {
    "inputs_N"          : 2,       
    "layers"            : [ (100, network.ReLU),(1, network.Sigmoid)],
    "weights_L"         : -0.1,     
    "weights_H"         : 0.1,      
    "save"              : False, #haven't implement this one yet  
    "loss"              : network.CrossEntropyLoss, 
    "learning_C"        : [],
    "testing_C"         : []
                   }

def replace_value_with_definition(key_to_find, definition):
    for key in initial_setting.keys():
        if key == key_to_find:
            initial_setting[key] = definition           

dataset = data.load()
features = dataset["train_data"]
targets = dataset["train_labels"]
test_features = dataset["test_data"]
test_targets = dataset["test_labels"]
m,n = features.shape
replace_value_with_definition("inputs_N",n)
replace_value_with_definition("weights_L",-1.0/n)
replace_value_with_definition("weights_H",1.0/n)
print initial_setting
NN = network.NetworkFrame(initial_setting)
features_normalized,mean,std = data.normalize(features)
test_normalized,_,_ = data.normalize(test_features,mean,std)
NN.Train(features_normalized, targets, test_normalized, test_targets, 10e-5, 100, 200, 0.001)
learning_record = NN.GetLearingRecord()
indecs = [x[0] for x in learning_record]
errors = [x[1] for x in learning_record]
testing_record = NN.GetTestingRecord()
testing_errors = [x[1] for x in testing_record]
visualization.LossCurve(indecs,errors,testing_errors)

