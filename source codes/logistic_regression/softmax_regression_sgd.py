import numpy as np


def softmax(scores):
    e = np.exp(scores)
    s = e.sum(axis=1)
    for i in range(len(s)):
        e[i] /= s[i]
    return e


class SoftmaxRegression:
    def __init__(self):
        self.W = None

    def fit(self, X, y, eta_0=50, eta_1=100, N=1000):
        m, n = X.shape
        m, k = y.shape
        W = np.zeros([n, k])
        self.W = W
        for t in range(N):
            i = np.random.randint(m)
            x = X[i].reshape(1, -1)
            proba = softmax(x.dot(W))
            g = x.T.dot(proba - y[i])
            W = W - eta_0 / (t + eta_1) * g
            self.W += W
        self.W /= N

    def predict_proba(self, X):
        return softmax(X.dot(self.W))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
