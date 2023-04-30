import numpy as np


class LinearRegressionMBGD:
    def __init__(self):
        self.w = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, eta_0=10, eta_1=50, N=10000, B=100):
        m, n = X.shape
        w = np.zeros((n, 1))
        self.w = w
        for t in range(N):
            batch = np.random.randint(low=0, high=m, size=B)
            X_batch = X[batch].reshape(B, -1)
            y_batch = y[batch].reshape(B, -1)
            e = X_batch.dot(w) - y_batch
            g = 2 * X_batch.T.dot(e) / B  # 小批量梯度计算公式跟随机梯度下降的公式基本是一样的
            w = w - (eta_0 / (t + eta_1)) * g  # w的更新公式有点不同
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
