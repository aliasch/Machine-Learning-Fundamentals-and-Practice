import numpy as np


class LinearRegressionGD:
    def __init__(self):
        self.w = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, eta, epsilon):
        m, n = X.shape
        w = np.zeros((n, 1))
        while True:
            e = X.dot(w) - y
            g = 2 * X.T.dot(e) / m
            w = w - eta * g
            if np.linalg.norm(g, 2) < epsilon:
                break
        self.w = w
        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]
        return self

    def predict(self, X):
        return X.dot(self.w)

    # 计算均方误差MSE
    def mse(self, y_true, y_pred):
        return np.average((y_true - y_pred) ** 2, axis=0)

    # 计算决定系数R^2
    def r2(self, y_true, y_pred):
        numerator = (y_true - y_pred) ** 2
        denominator = (y_true - np.average(y_true, axis=0)) ** 2
        return 1 - numerator.sum(axis=0) / denominator.sum(axis=0)
