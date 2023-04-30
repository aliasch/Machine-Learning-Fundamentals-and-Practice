import numpy as np


class LinearRegression:
    def __init__(self):
        self.w = None
        self.coef_ = None
        self.intercept_ = None

    # 根据正规方程拟合出最优的参数w，X为训练样本，为一个mxn的矩阵，y为样本标签，是一个mx1的矩阵
    def fit(self, X, y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]
        return self

    # 对新数据进行预测
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
