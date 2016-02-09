'''
Created on Feb 7, 2016

@author: Wuga
'''
import numpy as np
import scipy as sp

def CrossEntropyLoss(y,z,deriv=False):
    epsilon = 1e-15
    z=sp.maximum(z,epsilon)
    z=sp.minimum(z,1-epsilon)
    
    if deriv:            
        return -1.0/len(y)*(np.multiply(y,1/z)+np.multiply((1-y),1/(1-z)))
    else:
        return -1.0/len(y)*np.sum((np.multiply(y,np.log(z))+np.multiply((1-y),np.log(1-z))),axis=0)
    
# y=np.array([1,0,0,1])
# z=np.array([0.9,0.1,0.11,0.93])
# print (np.multiply(y,np.log(z))+np.multiply((1-y),np.log(1-z)))