'''
Created on 2-14-2016

@author: Wuga
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)

plt.plot(x, x)
plt.plot(x, 2 * x)
plt.plot(x, 3 * x)
plt.plot(x, 4 * x)
plt.show()