import numpy as np


class PCA:
    def __init__(self, n_components):
        self.d = n_components
        self.W = None
        self.mean = None

    def fit_transform(self, X):
        self.mean = X.mean(axis=0)
        X = X - self.mean
        eigen_values, eigen_vectors = np.linalg.eig(X.T.dot(X))
        n = len(eigen_values)
        pairs = [(eigen_values[i], eigen_vectors[:, i]) for i in range(n)]
        pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        self.W = np.array([pairs[i][1] for i in range(self.d)]).T
        return X.dot(self.W)

    def inverse_transform(self, Z):
        return Z.dot(self.W.T) + self.mean
