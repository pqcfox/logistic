import numpy as np


def optimize(cost_func, grad_func):
    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hypothesis(X, theta):
    return sigmoid(np.dot(theta, np.transpose(X)))
