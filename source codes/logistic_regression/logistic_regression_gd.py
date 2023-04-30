import numpy as np


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


class LogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, eta=0.1, N=1000):
        m, n = X.shape
        w = np.zeros((n, 1))
        for t in range(N):
            h = sigmoid(X.dot(w))
            g = 1.0 / m * X.T.dot(h - y)
            w = w - eta * g
        self.w = w

    def predict_proba(self, X):
        return sigmoid(X.dot(self.w))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)