'''
Created on Feb 6, 2016

@author: Wuga
'''

import numpy as np
def normalize(features, mean = [], std = []):
    if mean == []:
        mean = np.mean(features, axis = 0)
        std = np.std(features, axis = 0)
    print std
    print std[:,None]
    new_feature = (features.T - mean[:,None]).T
    new_feature = (new_feature.T / std[:,None]).T
    new_feature[np.isnan(new_feature)]=0
    print new_feature
    return new_feature, mean, std
 
    
        
