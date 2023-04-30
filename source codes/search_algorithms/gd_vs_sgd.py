import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression


class LinearRegressionGD:
    def __init__(self):
        self.w = None
        self.W = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, eta, N=1000):
        m, n = X.shape
        w = np.zeros((n, 1))
        self.W = np.zeros((N, 2))
        for t in range(N):  # 梯度下降每次循环的计算量大，但总体的循环次数N小
            # 有2个特征，所以权值w有两个分量，w[0],w[1]，这里记录N次循环过程中
            # w的值得变化细节，然后以w[0]为横坐标，w[1]为纵坐标话出图像，就可知道，w的优化过程
            self.W[t][0] = w[0]
            self.W[t][1] = w[1]
            e = X.dot(w) - y
            g = 2 * X.T.dot(e) / m  # 这里是所有样本，相差了m倍
            w = w - eta * g
        self.w = w
        return self

    def predict(self, X):
        return X.dot(self.w)


class LinearRegressionSGD:
    def __init__(self):
        self.w = None
        self.W = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, eta_0=10, eta_1=50, N=3000):
        m, n = X.shape
        w = np.zeros((n, 1))
        self.w = w
        self.W = np.zeros((N, 2))
        for t in range(N):
            self.W[t][0] = w[0]  # 记录权值w的变化，用于画图刻画w的轨迹
            self.W[t][1] = w[1]
            i = np.random.randint(m)
            x = X[i].reshape(1, -1)  # 为了计算方便，将第i个样本X[i]转换成列向量
            e = x.dot(w) - y[i]  # x^T * w -y_{i}，这里是一个样本
            gradient = 2 * e * x.T
            w = w - eta_0 * gradient / (t + eta_1)
            self.w += w
        self.w /= N
        return self

    def predict(self, X):
        return X.dot(self.w)


X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, bias=0, random_state=0)
y = y.reshape(-1, 1)
model = LinearRegressionGD()
model.fit(X, y, eta=0.01, N=3000)
plt.scatter(model.W[:, 0], model.W[:, 1], s=5)
plt.plot(model.W[:, 0], model.W[:, 1], 'y')
print(model.w)
model = LinearRegressionSGD()
model.fit(X, y)
plt.scatter(model.W[:, 0], model.W[:, 1], s=15)
plt.plot(model.W[:, 0], model.W[:, 1], 'b')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
print(model.w)
plt.show()
