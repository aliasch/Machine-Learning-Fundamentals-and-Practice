import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from clustering.agglomerative_clustering import AgglomerativeClustering
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def plt3D(X, centers, c, title):
    ax = plt.axes(projection='3d', elev=48, azim=134)
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=c, cmap='plasma')
    if centers is not None:
        ax.scatter3D(np.array(centers)[:, 0], np.array(centers)[:, 1], np.array(centers)[:, 2], c='b', s=180,
                     marker='*')
    plt.title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]
plt.figure(1)
plt3D(X, None, y, "鸢尾花的真实分类")
model = AgglomerativeClustering(n_clusters=3)
centers, assignments = model.fit_transform(X)
plt.figure(2)
plt3D(X, centers, assignments, "鸢尾花经过合并算法进行聚类")
plt.show()
