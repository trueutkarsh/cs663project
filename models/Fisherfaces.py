# Correlation Model

import os
import time
import numpy as np
from sklearn.decomposition import PCA

#import Base

def find_basis(X, n_c):
    # return top n_c orthogonal eigen vectors
    r =  np.linalg.matrix_rank(X)
    k = min(r, n_c)
    q, r = np.linalg.qr(X)
    return q[:,:k]

class Fisherfaces:

    def __init__(self, X, Y):
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
        self.n_components = 100
        self.w = None
        self.model = PCA()
        self.k = self.numClasses - 1
        

    def fit(self):
        start = time.time()
        #write function definition here
        #form basis for each class of images
        #improvise this
        
        # self.sampleBasis = np.zeros((self.numClasses, self.imgShape, self.n_components))
        
        # #form basis 
        # for c in range(self.numClasses):
        #     self.sampleBasis[c, :, :] = find_basis(self.trainX[c].transpose(), self.n_components)

        self.trainX = self.trainX.reshape((np.prod(self.trainX.shape[:2]), self.trainX.shape[2]))
        # self.trainY = self.trainY.reshape((np.prod(self.trainY.shape[:2])))


        # reduce the dimensionality to N - c 
        self.imgShape =  self.n_components
        self.model = PCA(n_components = self.imgShape)
        self.trainX= self.model.fit_transform(self.trainX)   

        # compute st
        w =  (self.trainX - self.trainX.mean(axis = 0))
        #print("w.shape", w.shape)
        s_t = (w.T).dot(w) 
        #print("st shape", s_t.shape)
        
        va, ve = np.linalg.eig(s_t)
        #print(va.shape, "eig", ve.shape)
        id = va.argsort()[::-1]   

        w_pca = ve[:, id]
        #print(w_pca.shape)
        w_pca = w_pca[:, :self.k]

        #print("w_pca shape", w_pca.shape)
        
        # reshape again to calculate Sw and Sb
        self.trainX = self.trainX.reshape((self.numClasses, self.numImages, self.imgShape ))
        

        mean_sample  = np.apply_over_axes(np.mean, self.trainX, [1])
        mean_global = np.apply_over_axes(np.mean, self.trainX, [0, 1])

        mean_global = np.squeeze(mean_global)
        mean_sample = np.squeeze(mean_sample)


        # compute sw/sb
        self.sw = np.zeros((self.imgShape, self.imgShape))
        self.sb = np.zeros((self.imgShape, self.imgShape))
        for s in range(self.numClasses):
            for i in range(self.numImages):
                self.sw = self.sw + (self.trainX[s][i] - mean_sample[s]).dot((self.trainX[s][i] - mean_sample[s]).transpose())
            self.sb = self.sb + self.numImages*(mean_sample[s] - mean_global).dot((mean_sample[s] - mean_global).transpose())

        print("sb shape", )


        #compte new sb
        self.sw = (w_pca.T).dot(self.sw).dot(w_pca)
        self.sb = (w_pca.T).dot(self.sb).dot(w_pca)

        [V, D] = np.linalg.eig( ( np.linalg.inv(self.sw)).dot(self.sb) )

        idx = V.argsort()[::-1]   

        # k = max(0, min(self.numClasses - 1, self.k))
        k = self.k
        # choose top k if needed
        eigVe = D[:, idx]
        eigVa = V[idx]

        # w formed
        self.w = w_pca.dot(eigVe[:, :k])
        #convert to n x d format
        self.trainX = self.trainX.reshape((np.prod(self.trainX.shape[:2]), self.trainX.shape[2]))
        self.trainY = self.trainY.reshape((np.prod(self.trainY.shape[:2])))

        train_c = (self.trainX).dot(self.w)
        self.trainX = train_c.dot(self.w.T)

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
        
        value = self.model.transform(value)

        k = self.w.shape[1]
        
        value = (value).dot(self.w)
        # print(coeffs.shape, value.shape)
        value = (self.w).dot(value.T)

        idx = np.abs(self.trainX - value.T).sum(axis = 1).argmin()
        
        return idx/self.numImages

    def score(self, X, Y):
        #X = (X - X.mean(axis = 0))/X.std(axis = 0)
        result = self.predict(X)
        diff = np.rint(result - Y)
        self.accuracy = float((result.shape[0] - np.count_nonzero(diff)))/float(result.shape[0])
        return self.accuracy, self.fitTime

if __name__ == '__main__':
    c = LDA(None, None)
