import numpy as np
import linear_regression as lib
import matplotlib.pyplot as plt

data = np.array([
[280.0,	3.909,	9.43],
[281.5,	5.119,	10.36],
[337.4,	6.666,	14.50],
[404.2,	5.338,	15.75],
[402.1,	4.321,	16.78],
[452.0,	6.117,	17.44],
[431.7,	5.559,	19.77],
[582.3,	7.920,	23.76],
[596.6,	5.816,	31.61],
[620.8,	6.113,	32.17],
[513.6,	4.258,	35.09],
[606.9,	5.591,	36.42],
[629.0,	6.675,	36.58],
[602.7,	5.543,	37.14],
[656.7,	6.933,	41.30],
[998.5,	7.638,	45.62],
[877.6,	7.752,	47.38]])

X = data[:, 1:]
y = data[:, 0]

def process_features(X):
    m, n = X.shape  # 获取训练样本的大小
    X = np.c_[np.ones((m, 1)), X]  # np.c_在这里的作用是在矩阵X的前面拼接一个mx1的全1向量，构成一个新的X
    return X


# plt.scatter(X, y)
X = process_features(X)
# 调用模型，拟合出适当的参数w
model = lib.LinearRegression()
model.fit(X, y)
print(f"b = {model.intercept_}, w = {model.coef_}")

