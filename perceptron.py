from random import choice
import numpy as np
from numpy import array, dot, random


class Perceptron(object):
    def __init__(self):
        self.w             = []
        self.errors        = []
        self.rate           = 0.05 # learning rate
        self.epochs        = 500 # epochs
        self.step_function = lambda x: 0 if x < 0 else 1


    def train(self, X, y):
        # Add bias column to X
        _X     = np.c_[X, np.ones(X.shape[0])]
        x_len  = _X.shape[1]
        # Calculate random weights
        self.w = random.rand(x_len)

        for i in xrange(self.epochs):
            rand_i      = random.randint(x_len)
            x, expected = _X[rand_i], y[rand_i]
            result      = dot(self.w, x)
            error       = expected - self.step_function(result)
            self.errors.append(error)
            self.w += self.rate * error * x


    def predict(self, X):
        _X     = np.c_[X, np.ones(X.shape[0])]
        for x in _X:
            result = dot(x, self.w)
            print("{}: {} -> {}".format(x[:2], result, self.step_function(result)))
