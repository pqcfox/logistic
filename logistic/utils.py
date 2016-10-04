import numpy as np


def gradient_descent(grad_func, m, alpha=1e-2, beta=1e-7, epsilon=1e-10):
    theta = 2 * (np.random.rand(m) - 0.5) * beta
    while True:
        delta = alpha * grad_func(theta)
        theta -= delta
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
