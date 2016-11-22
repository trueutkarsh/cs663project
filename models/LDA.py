# Correlation Model

import os
import time
import numpy as np
#import Base

def find_basis(X, n_c):
    # return top n_c orthogonal eigen vectors
    r =  np.linalg.matrix_rank(X)
    k = min(r, n_c)
    q, r = np.linalg.qr(X)
    return q[:,:k]

class LDA:

    def __init__(self, X, Y, n_components = 3):
        self.trainX = X
        self.trainY = Y
        self.numClasses = X.shape[0]
        self.numImages = X.shape[1]
        self.imgShape = X.shape[2]
        self.fitted = False
        self.accuracy = 0.0
        self.fitTime = 0.0
        self.predictTime = 0.0
        self.predict_result = None
        self.sampleBasis = None
        self.sb = None
        self.sw = None
        if n_components > self.numClasses -1 :
            raise OSError("Invalid n_components")
        self.n_components = n_components
        self.w = None

    def fit(self):
        start = time.time()
        #write function definition here
        #form basis for each class of images
        #improvise this
        
        # self.sampleBasis = np.zeros((self.numClasses, self.imgShape, self.n_components))
        
        # #form basis 
        # for c in range(self.numClasses):
        #     self.sampleBasis[c, :, :] = find_basis(self.trainX[c].transpose(), self.n_components)

                

        mean_sample  = np.apply_over_axes(np.mean, self.trainX, [1, 2])
        mean_global = np.apply_over_axes(np.mean, self.trainX, [0, 1, 2])

        # compute sw/sb
        self.sw = np.zeros((self.imgShape, self.imgShape))
        self.sb = np.zeros((self.numImages, self.numImages))
        for s in self.numClasses:
            for i in self.numImages:
                sw = sw + (self.trainX[s][i] - mean_sample[s]).dot((self.trainX[s][i] - mean_sample[s]).transpose())
            sb = sb + self.numImages*(mean_sample[s] - mean_global).dot((mean_sample[s] - mean_global).transpose())

        
        [V, D] = np.linalg.eig( np.inv(sw)*(sb) )

        idx = V.argsort()[::-1]   

        k = max(0, min(self.numClasses - 1, self.n_components))
        # choose top k if needed
        eigVe = D[:, idx]
        eigVa = V[idx]

        # w formed
        self.w = eigVe[:, :k]

        end = time.time()
        self.fitTime = end - start
        self.fitted = True
    

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
        
        
        coeffs = self.sampleBasis.reshape((self.numClasses, self.n_components, self.imgShape)).dot(value)
        coeffs = coeffs.reshape(self.numClasses, self.n_components)

        projection = np.zeros((self.numClasses, self.imgShape))
        for x in range(self.numClasses):
            projection[x,:] = self.sampleBasis[x].dot(coeffs[x, :])


        idx = np.abs(projection - value).sum(axis = 1).argmin()
        
        return idx

    def score(self, X, Y):
        #X = (X - X.mean(axis = 0))/X.std(axis = 0)
        result = self.predict(X)
        diff = np.rint(result - Y)
        self.accuracy = float((result.shape[0] - np.count_nonzero(diff)))/float(result.shape[0])
        return self.accuracy, self.fitTime

if __name__ == '__main__':
    c = LDA(None, None)
