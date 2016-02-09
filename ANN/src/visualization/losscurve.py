'''
Created on Feb 8, 2016

@author: Wuga
'''

import matplotlib.pyplot as plt

ax = plt.subplot(111)

def LossCurve(epoch, error):
    plt.figure(1)
    plt.subplot(111)
    plt.plot(epoch, error)
    plt.yscale('linear') #or log
    plt.title('Learning Error v.s Epoch')
    plt.grid(True)
    plt.show()
    
# X=np.array([[1,1],[2,4],[3,9]])
# LossCurve(X[:,0], X[:,1])