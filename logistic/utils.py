import numpy as np


def gradient_descent(cost_func, grad_func, m, alpha=1e-5, beta=1e-7, epsilon=1e-6):
    theta = 2 * (np.random.rand(m) - 0.5) * beta
    cost = cost_func(theta)
    while True:
        delta = alpha * grad_func(theta)
        print("WAT")
        print(cost)
        theta -= delta
        new_cost = cost_func(theta)
        print(np.abs(new_cost - cost))
        if np.abs(new_cost - cost) < epsilon:
            break
        cost = new_cost
    return theta


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hypothesis(X, theta):
    X, theta = np.asarray(X), np.asarray(theta)
    if X.size == 0 or theta.size == 0:
        return np.array([])  # np.dot([], []) returns 0
    return sigmoid(np.dot(theta, np.transpose(X)))
