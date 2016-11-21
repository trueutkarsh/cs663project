# Base model class to declare all models for face recognition
import os

class Model:

    def __init__(self, name):
        self.name = name
        self.fitted = False

    def fit(X, Y):
        pass

    def predict(X):
        pass



if __name__ == '__main__':
    base = Model(name="Base")

                   
