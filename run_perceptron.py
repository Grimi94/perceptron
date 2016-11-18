from numpy import array
from perceptron import Perceptron
import sys


if __name__ == "__main__":
    X = []
    y = []

    for line in sys.stdin:
        content = line.strip("\n").split(",")
        X.append(map(int, content[:-1]))
        y.append(map(int, content[-1]))

    X = array(X)
    y = array(y)

    p = Perceptron()
    p.train(X, y)
    p.predict(X)

    # Print weights
    # print p.w
