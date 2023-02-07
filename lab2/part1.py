import numpy as np
import matplotlib.pyplot as plt
def makedataset(stepsize,length):
    train_x = np.arange(0,length,stepsize)
    test_x = np.arange(0.05,length,stepsize)
    y_train= np.sin(train_x)
    y_test=np.sin(test_x)
    return train_x,y_train,test_x,y_test

def gaussianrbfmatrix(x,centres,sigma):
    return np.exp(-np.power(x-centres,2)/(2*np.power(sigma,2)))
def batchlearning()

