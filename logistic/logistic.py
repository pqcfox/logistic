import numpy as np

from logistic.utils import optimize, hypothesis


class Logistic:
    def fit(self, X, y):
        cost_func = self.cost(X, y)
        grad_func = self.grad(X, y)
        self.theta = optimize(cost_func, grad_func)
        return self

    def predict(self, X):
        return np.dot(self.theta, np.transpose(X)) > 0.5

    @staticmethod
    def cost(X, y):
        def f(theta):
            pos = y * np.log10(hypothesis(X, theta))
            neg = (1 - y) * np.log10(1 - hypothesis(X, theta))
            return -sum(pos + neg) / y.shape[0]
        return f

    @staticmethod
    def grad(X, y):
        def f(theta):
            inside = np.dot(hypothesis(X, theta) - y, X)
            return np.transpose(inside) / y.shape[0]
        return f
