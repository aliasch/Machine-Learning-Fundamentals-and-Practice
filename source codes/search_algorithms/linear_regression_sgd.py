import numpy as np


class LinearRegressionSGD:
    def __init__(self):
        self.w = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, eta_0=10, eta_1=50, N=10000):
        m, n = X.shape
        w = np.zeros((n, 1))
        self.w = w
        for t in range(N):
            i = np.random.randint(m)
            x = X[i].reshape(1, -1)
            e = x.dot(w) - y[i]
            # 特别注意，这里的e是一个数，所以这里的乘法是数*向量，不是矩阵乘法
            g = 2 * e * x.T
            w = w - eta_0 * g / (t + eta_1)
            self.w += w
        self.w /= N
        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]
        return self

    def predict(self, X):
        return X.dot(self.w)

    def mse(self, y_true, y_pred):
        return np.average((y_true - y_pred) ** 2, axis=0)

    def r2(self, y_true, y_pred):
        numerator = (y_true - y_pred) ** 2
        denominator = (y_true - np.average(y_true, axis=0)) ** 2
        return 1 - numerator.sum(axis=0) / denominator.sum(axis=0)
