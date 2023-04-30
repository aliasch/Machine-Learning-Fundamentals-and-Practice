import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from kernel_svm import KernelSVM


def rbf_kernel(x1, x2):
    sigma = 1.0
    return np.exp(-np.linalg.norm(x1 - x2, 2) ** 2 / sigma)


iris = datasets.load_iris()
X = iris["data"][:, (1, 3)]  # 获得所有样本的第1，第3列特征
y = 2 * (iris["target"] == 1).astype(int).reshape(-1, 1) - 1
plt.subplot(1, 3, 1)
plt.xlim(([0.5, 5.5]))
plt.ylim(([0, 3]))
plt.title('鸢尾花训练样本')
plt.scatter(X[:, 0][y[:, 0] == 1], X[:, 1][y[:, 0] == 1], label='变色鸢尾花')
plt.scatter(X[:, 0][y[:, 0] == -1], X[:, 1][y[:, 0] == -1], label='非变色鸢尾花')
plt.legend()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=60)
plt.subplot(1, 3, 2)
plt.title('训练样本分类结果')
plt.scatter(X_train[:, 0][y_train[:, 0] == 1], X_train[:, 1][y_train[:, 0] == 1], label='变色鸢尾花')
plt.scatter(X_train[:, 0][y_train[:, 0] == -1], X_train[:, 1][y_train[:, 0] == -1], label='非变色鸢尾花')
# model = KernelSVM(kernel=rbf_kernel)
model = KernelSVM(kernel=None)
model.fit(X_train, y_train)
x0s = np.linspace(0.5, 5.5, 100)
x1s = np.linspace(0, 3, 100)
x0, x1 = np.meshgrid(x0s, x1s)
W = np.c_[x0.ravel(), x1.ravel()]  # ravel方法将矩阵降维为向量
u = model.predict(W).reshape(x0.shape)  # W为测试数据
plt.contourf(x0, x1, u, alpha=0.2)
plt.legend()
plt.subplot(1, 3, 3)
plt.title('测试样本的分类结果')
plt.contourf(x0, x1, u, alpha=0.2)
plt.scatter(X_test[:, 0][y_test[:, 0] == 1], X_test[:, 1][y_test[:, 0] == 1], marker='*', label='变色鸢尾花')
plt.scatter(X_test[:, 0][y_test[:, 0] == -1], X_test[:, 1][y_test[:, 0] == -1], marker='x', label='非变色鸢尾花')
plt.legend()
plt.tight_layout()
plt.show()
