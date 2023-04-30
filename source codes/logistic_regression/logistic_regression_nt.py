import numpy as np


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


class LogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, N=1000):
        m, n = X.shape
        w = np.zeros((n, 1))
        for t in range(N):
            pred = sigmoid(X.dot(w))  # h_w(x)
            g = 1.0 / m * X.T.dot(pred - y)  # F(W)的梯度
            pred = pred.reshape(-1)
            D = np.diag(pred * (1 - pred))  # h_w(x)(1-h_w(x))再将这些值作为矩阵的对角元素
            H = 1.0 / m * (X.T.dot(D)).dot(X)  # F(W)的Hessian方阵
            w = w - np.linalg.inv(H).dot(g)  # w=w-F'(W)/F''(W) 多元牛顿迭代
        self.w = w

    def predict_proba(self, X):
        return sigmoid(X.dot(self.w))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int)
