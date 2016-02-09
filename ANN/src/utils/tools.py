'''
Created on Feb 6, 2016

@author: Wuga
'''
import numpy as np

def add_ones(A):
    return np.hstack(( np.ones((A.shape[0],1)), A ))