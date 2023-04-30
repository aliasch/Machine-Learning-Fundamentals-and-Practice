import matplotlib.pyplot as plt
import numpy as np
import linear_regression.linear_regression as lib

np.random.seed(0)


def generate_samples(m):
    X = np.random.uniform(-1, 1, (m, 1))
    fx = []
    for i in range(len(X)):
        if X[i][0] < -0.5:
            fx.append(-1)
        elif -0.5 <= X[i][0] < 0:
            fx.append(1)
        elif 0 <= X[i][0] < 0.5:
            fx.append(-0.5)
        else:
            fx.append(0.5)
    y = np.random.normal(fx, 0.05, m)
    return X, y


def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


X, y = generate_samples(50)
# 线性回归的X需要作一些特殊处理，把偏置项b放进去，全1列
X = process_features(X)
model = lib.LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], y_pred[:])
plt.show()
