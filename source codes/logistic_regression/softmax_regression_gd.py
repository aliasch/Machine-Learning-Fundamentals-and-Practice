import numpy as np


class SoftmaxRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, eta=0.1, N=5000):
        m, n = X.shape
        m, k = y.shape
        w = np.zeros([n, k])
        for t in range(N):
            proba = self.softmax(X.dot(w))  # proba 就是h_{w}(x)
            g = X.T.dot(proba - y) / m  # g就是f(w)的梯度
            w = w - eta * g   # 进行梯度下降算法的迭代
        self.w = w

    @staticmethod
    def softmax(x):  # x会传入X@w
        e = np.exp(x)
        s = e.sum(axis=1)
        for i in range(len(s)):
            e[i] /= s[i]
        return e

    def predict_proba(self, X):  # 进行概率值预测
        return self.softmax(X.dot(self.w))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)  # 获取概率值最大的分量下标，将其作为判别的类别
