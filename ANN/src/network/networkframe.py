'''
Created on Feb 6, 2016

@author: Wuga
'''
import numpy as np
import utils
import random
from activations import ReLU,Sigmoid

#===============================================================================
# This is a framework of neural network
#
# When initializing the class will:
# 1. compute number of weights needed based on input dimensions and layer info
# 2. convert random initialized weights into matrixes that corresponding to size
#    of each layer.
#
# FeedForward function accept a matrix (number of data \times number of dimensi-
# on) as input, and output a vector(number of data) of prediction as output.
# when backpropagation is switched on, it also trace intermediate results and 
# derivative of each layer.
#
# BackPropagation function update weights of the network. Nothing funcy but sim-
# ply chain rule.
#
# Train function wraps forward and backward function with epoch loops that upda-
# te weights iteratively. You can allows data permutation for each loop by unco-
# mment the block that indicate permutation.
#===============================================================================

class NetworkFrame:
    
    def __init__(self, settings):
        self.__dict__.update(settings)
        self.weights_N=(self.inputs_N+1)*self.layers[0][0]+\
            sum( (self.layers[index][0]+1)*layer[0] for index,layer in enumerate(self.layers[1:]))
        self.weights_Matrics = self.ReshapeWeights(self.RandomInitialWeights())
        
    def RandomInitialWeights(self):
        return np.random.uniform(self.weights_L,self.weights_H,self.weights_N)
    
    def ReshapeWeights(self, weights_flat):
        head = 0;
        tail = 0;
        weights_matrics = []
        
        previous_node_N = self.inputs_N + 1
        
        for current_node_N, node_T in self.layers:
            head = tail
            tail += previous_node_N * current_node_N
            weights_matrics.append(weights_flat[head:tail].reshape(previous_node_N,current_node_N))
            previous_node_N = current_node_N + 1
        return weights_matrics
    
    def GetWeights(self):
        return self.weights_N
            
    def FeedForward(self, inputs, backpropagation = False):
        
        output = inputs
        
        if backpropagation:
            outputs = [ output ]
            derivates = []
        
        for index, weight_Matrix in enumerate(self.weights_Matrics):
            signal = weight_Matrix[0:1,:]+np.dot(output, weight_Matrix[1:,:])
            output = self.layers[index][1](signal)
            
            if backpropagation:
                outputs.append(output)
                derivates.append(self.layers[index][1](signal,True).T)
        
        if backpropagation:
            return outputs, derivates
        else:
            return output
        
    def BackPropagation(self, outputs, derivates, training_targets, learning_rate):
        output = outputs[-1]
        #error = self.loss(training_targets, output, True).T
        error = (training_targets-output).T
        delta = error*derivates[-1]
        
        for i in range( len(self.layers) )[::-1]:
            delta_w = learning_rate*np.dot(delta, utils.add_ones(outputs[i])).T
            if i != 0:
                delta = np.dot(self.weights_Matrics[i][1:,:], delta)*derivates[i-1]
            self.weights_Matrics[i]+=delta_w
        
    def Train(self, training_features, training_targets, learning_rate = 0.05, minibatch_size = 10, max_epoch = 1000, error_threshold = 1e-3):
        epoch = 0
        error = 1
        features = np.copy(training_features)
        targets = np.copy(training_targets)
        
        while error > error_threshold and epoch < max_epoch:
            epoch += 1
            perm = range(len(features))
            random.shuffle(perm)
#             features = features[perm]
#             targets = targets[perm]
            for i in range(len(features)/minibatch_size):
                mini_features = features[i*minibatch_size:(i+1)*minibatch_size]
                mini_targets = targets[i*minibatch_size:(i+1)*minibatch_size]
                outputs, derivates = self.FeedForward(mini_features, True)
                error = self.loss(mini_targets, outputs[-1], deriv=False)
                self.BackPropagation(outputs, derivates, mini_targets, learning_rate)
            train_output = self.FeedForward(features, False)
            train_error = self.loss(targets, train_output, deriv=False)
            self.learning_C.append([epoch, train_error[0]])
            print "* Epoch %d : Error %f." %(epoch,train_error)
        
        if epoch%1000==0:
        # Show the current training status
            print "* current network error (Cross-Entropy):", error
        print "* Trained for %d epochs." % epoch 
        
    def Test(self, test_features):
        return self.FeedForward(test_features, False)        
        
    def GetLearingRecord(self):
        return self.learning_C
        