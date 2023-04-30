import numpy as np
class RidgeRegression:
    def __init__(self, Lambda):
        self.Lambda = Lambda
        self.w = None
        self.coef_ = None
        self.intercept_ = None
    def fit(self, X, y):
        m, n = X.shape
        r = np.diag(self.Lambda * np.ones(n))
        self.w = np.linalg.inv(X.T.dot(X) + r).dot(X.T).dot(y)
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
