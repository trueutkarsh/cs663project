# Correlation Model

import os
import time
import numpy as np
from sklearn.decomposition import PCA
#import Base

class EigenFaces:

    def __init__(self, X, Y, n_components = 5):
        self.trainX = X
        self.trainY = Y
        self.numClasses = X.shape[0]
        self.numImages = X.shape[1]
        self.fitted = False
        self.accuracy = 0.0
        self.fitTime = 0.0
        self.predictTime = 0.0
        self.predict_result = None
        self.model = PCA(n_components=n_components)


    def fit(self):
        start = time.time()
        #write function definition here
        self.trainX = self.trainX.reshape((np.prod(self.trainX.shape[:2]), self.trainX.shape[2]))
        self.trainY = self.trainY.reshape((np.prod(self.trainY.shape[:2])))
        self.trainX = (self.trainX - self.trainX.mean(axis = 0))/self.trainX.std(axis = 0)

        #fit model
        self.trainX = self.model.fit_transform(self.trainX)
        end = time.time()
        self.fitTime = end - start
        self.fitted = True
        pass

    def predict(self, X):
        if not self.fitted:
            raise OSError("Fit a dataset first")
        else:
            start = time.time()
            self.predict_result = np.apply_along_axis(self._findNearestIndex, 1, X)
            end = time.time()
            self.predictTime = end - start
            return self.predict_result
    
    def _findNearestIndex(self, value):
        value = self.model.transform(value)
        idx = np.abs(self.trainX - value).sum(axis = 1).argmin()
        idx = int(idx / self.numImages)
        return idx

    def score(self, X, Y):

        X = (X - X.mean(axis = 0))/X.std(axis = 0) #normalize the images
        result = self.predict(X)
        diff = np.rint(result - Y)
        self.accuracy = float((result.shape[0] - np.count_nonzero(diff)))/float(result.shape[0])
        return self.accuracy, self.fitTime

if __name__ == '__main__':
    c = EigenFaces(None, None)
