from numpy import array
from perceptron import Perceptron


if __name__ == "__main__":
    p = Perceptron()
    p.train(array(X_and), y_and)
    p.predict(array(X_and))
    print p.w
