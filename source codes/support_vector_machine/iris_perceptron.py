import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()
X = iris["data"][:, (0, 1)]
y = 2 * (iris["target"] == 0).astype(int) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
plt.subplot(1, 2, 1)
plt.axis([4, 8, 1.5, 5])
plt.scatter(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], marker="*")
plt.scatter(X_train[:, 0][y_train == -1], X_train[:, 1][y_train == -1], marker="+")
model = Perceptron()
model.fit(X_train, y_train)

x0 = np.linspace(4, 8, 200)
# line就是x1，因为感知机方程x^{T}w + b = 0,就是w1*x1 + w0 * x0 + b = 0
y0 = -model.w[0] / model.w[1] * x0 - model.b / model.w[1]
plt.plot(x0, y0)
plt.title("感知机划分训练数据集")
plt.xlabel("花萼长")
plt.ylabel("花萼宽")

plt.subplot(1, 2, 2)
plt.axis([4, 8, 1.5, 5])
plt.scatter(X_test[:, 0][y_test == 1], X_test[:, 1][y_test == 1])
plt.scatter(X_test[:, 0][y_test == -1], X_test[:, 1][y_test == -1])
x0 = np.linspace(4, 8, 200)
y0 = -model.w[0] / model.w[1] * x0 - model.b / model.w[1]
plt.plot(x0, y0)
plt.title("感知机划分测试数据集")
plt.xlabel("花萼长")
plt.ylabel("花萼宽")

plt.tight_layout()
plt.show()
