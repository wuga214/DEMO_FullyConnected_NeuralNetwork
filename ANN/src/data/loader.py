'''
Created on Jan 31, 2016

@author: Wuga
'''
import os

def unpickle(file_path):
    try:
        import cPickle as pickle
    except:
        import pickle
    
    fo = open(file_path, 'rb')
    data = pickle.load(fo)
    fo.close()
    return data

def load():
    datapath="data/cifar_2class_py2.p"
    script_dir = os.path.dirname(__file__)
    abs_data_path = os.path.join(script_dir, '..', '..',datapath)
    data=unpickle(abs_data_path)
    return data