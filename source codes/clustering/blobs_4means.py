import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from clustering.k_means import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
N_clusters = 4
X, y = make_blobs(n_samples=100, centers=N_clusters, random_state=0, cluster_std=0.8)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title("原始数据")
model = KMeans(n_clusters=N_clusters, max_iter=100)
centers, assignments = model.fit_transform(X)
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=assignments)
plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], c='b', s=180, marker='*')
plt.title("聚类的结果")
plt.show()
