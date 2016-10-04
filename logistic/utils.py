import numpy as np


def optimize(cost_func, grad_func):
    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hypothesis(X, theta):
    X, theta = np.asarray(X), np.asarray(theta)
    if X.size == 0 or theta.size == 0:
        return np.array([])  # np.dot([], []) returns 0
    return sigmoid(np.dot(theta, np.transpose(X)))
