import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from svm_smo import SVM

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()
X = iris["data"][:, (0, 1)]
y = 2 * (iris["target"] == 0).astype(int).reshape(-1, 1) - 1  # 将标签转为-1，+1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

model = SVM()
model.fit(X_train, y_train, N=10)
plt.subplot(1, 2, 1)
plt.axis([4, 8, 1.5, 5])
plt.scatter(X_train[:, 0][y_train[:, 0] == 1], X_train[:, 1][y_train[:, 0] == 1])
plt.scatter(X_train[:, 0][y_train[:, 0] == -1], X_train[:, 1][y_train[:, 0] == -1])

x0 = np.linspace(4, 8, 200)
y0 = -model.w[0] / model.w[1] * x0 - model.b / model.w[1]
plt.plot(x0, y0)
plt.title("支持向量机划分训练数据集")
plt.xlabel("花萼长")
plt.ylabel("花萼宽")

plt.subplot(1, 2, 2)
plt.axis([4, 8, 1.5, 5])
plt.scatter(X_test[:, 0][y_test[:, 0] == 1], X_test[:, 1][y_test[:, 0] == 1])
plt.scatter(X_test[:, 0][y_test[:, 0] == -1], X_test[:, 1][y_test[:, 0] == -1])
x0 = np.linspace(4, 8, 200)
y0 = -model.w[0] / model.w[1] * x0 - model.b / model.w[1]
plt.plot(x0, y0)
plt.title("支持向量机划分测试数据集")
plt.xlabel("花萼长")
plt.ylabel("花萼宽")
plt.tight_layout()
plt.show()
