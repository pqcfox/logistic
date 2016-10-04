import numpy as np


def gradient_descent(grad_func, m, alpha=0.01, epsilon=0.0000001):
    theta = np.zeros(m)
    while True:
        delta = alpha * grad_func(theta)
        theta -= delta
        print(np.linalg.norm(delta))
        print(theta)
        if np.linalg.norm(delta) < epsilon:
            break
    return theta


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hypothesis(X, theta):
    X, theta = np.asarray(X), np.asarray(theta)
    if X.size == 0 or theta.size == 0:
        return np.array([])  # np.dot([], []) returns 0
    return sigmoid(np.dot(theta, np.transpose(X)))
