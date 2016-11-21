# This is the base class just to implement for training and test databases


# write code for generating training and test image
# check if already a hdf5 file not present

import os
import pandas as pd
import numpy as np
import math as mt
import random as rndm

import matplotlib.pyplot as plt
from scipy.misc import imread


def shufflesplice(l, n):
    rndm.shuffle(l)
    return l[:n]


baseAddress = "./data/CroppedYale/"
downsamplefactor = 2
numSamples = 11
numImages = 20
imgx = int(192/downsamplefactor)
imgy = int(168/downsamplefactor)
splitratio = 0.7
numTrain =int(mt.floor(numImages*splitratio))
numTest = numImages - numTrain
trainX = np.zeros((numSamples, numTrain, imgx*imgy))
trainY = np.zeros((numSamples, numTrain))
testX = np.zeros((numSamples , numTest, imgx*imgy))
testY = np.zeros((numSamples , numTest))

samplelist = shufflesplice(os.listdir(baseAddress), numSamples)

for c in range(0, numSamples):

    imglist = shufflesplice(os.listdir(os.path.join(baseAddress, samplelist[c])), numImages)

    for i in range(0, numTrain):
        imgpath = os.path.join(baseAddress, samplelist[c], imglist[i])
        #print(imgpath)
        #plt.imshow(np.array(imread(imgpath))[::downsamplefactor, ::downsamplefactor], cmap='Greys_r')
        trainX[c, i] = np.array(imread(imgpath))[::downsamplefactor, ::downsamplefactor].reshape((imgx*imgy))
        trainY[c, i] = c
        
    for i in range(numTrain, numImages):
        imgpath = os.path.join(baseAddress, samplelist[c], imglist[i])
        #print(imgpath)
        testX[c, i - numTrain] = np.array(imread(imgpath))[::downsamplefactor, ::downsamplefactor].reshape((imgx*imgy))
        testY[c, i - numTrain] = c        

print("Splitting data done !")