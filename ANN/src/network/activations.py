'''
Created on Feb 6, 2016

@author: Wuga
'''

#===============================================================================
# Two activation functions are available for you to choose
# But you can also implement others like tant function
#
# ReLU used here is modified one, which is more stable for preventing output 
# from exploration
#
# Sigmoid is classical nonlinear function used in neural network.
# No trick here, I only add a boundary to prevent unexpected overflow of output
#===============================================================================

import numpy as np

def ReLU(nets,derivative=False):
    output= np.maximum(nets,0)
    if derivative:
        derivate = output
        derivate[derivate != 0] = 1
        return derivate
    else:
        return output
    
def Sigmoid(nets, derivative=False):
    output = np.clip( nets, -500, 500 )
    output= 1.0/(1+np.exp(-output))
    if derivative:
        derivate=np.multiply(output,1-output)
        return derivate
    else:
        return output