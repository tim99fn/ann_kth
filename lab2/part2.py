import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import pandas as pd
def makesinedataset(stepsize,length):
    train_x = np.arange(0,length,stepsize)
    test_x = np.arange(0.05,length,stepsize)
    y_train= np.sin(train_x)
    y_test=np.sin(test_x)
    #plt.plot(train_x,y_train)
    return train_x,y_train,test_x,y_test
def makesquaredataset(stepsize,length):
    train_x,y_train,test_x,y_test=makesinedataset(stepsize,length)
    y_train[y_train>0]=1
    y_test[y_test>0]=1
    y_train[y_train<0]=-1
    y_test[y_test<0]=-1
    return train_x,y_train,test_x,y_test
def makesinedatasetwithnoise(stepsize,length):
    train_x = np.arange(0,length,stepsize)
    test_x = np.arange(0.05,length,stepsize)
    y_train= np.sin(train_x)
    y_test=np.sin(test_x)
    y_train=y_train+np.random.normal(0,0.1,len(y_train))
    y_test=y_test+np.random.normal(0,0.1,len(y_test))
    #plt.plot(train_x,y_train)
    return train_x,y_train,test_x,y_test
def makesquaredatasetwithnoise(stepsize,length):
    train_x,y_train,test_x,y_test=makesquaredataset(stepsize,length)
    y_train=y_train+np.random.normal(0,0.1,len(y_train))
    y_test=y_test+np.random.normal(0,0.1,len(y_test))
def importdata():
     train = pd.read_csv('ballist.dat')
     train=train.to_numpy()
     test = pd.read_csv('balltest.dat')
     test=test.to_numpy()
     x_train=train[:,0:2]
     y_train=train[:,2:4]
     
     