import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from clustering.k_means import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
iris = datasets.load_iris()
X = iris.data
y = iris.target
N_clusters = 3
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0][y[:] == 0], X[:, 1][y[:] == 0], c='b')
plt.scatter(X[:, 0][y[:] == 1], X[:, 1][y[:] == 1], c='y')
plt.scatter(X[:, 0][y[:] == 2], X[:, 1][y[:] == 2], c='k')
plt.title("原始数据")
model = KMeans(n_clusters=N_clusters, max_iter=100)
centers, assignments = model.fit_transform(X)
plt.subplot(1, 2, 2)
# 具有相同的assignments是同一类，颜色一样
plt.scatter(X[:, 0], X[:, 1], c=assignments)
# 绘制出3个聚类的中心点（质心）
plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], c='b', s=180, marker='*')
plt.title("聚类的结果")
plt.show()
