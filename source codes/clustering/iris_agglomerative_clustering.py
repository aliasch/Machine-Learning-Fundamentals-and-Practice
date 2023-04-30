import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from agglomerative_clustering import AgglomerativeClustering

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("鸢尾花的真实分类")
plt.xticks([])
plt.yticks([])
model = AgglomerativeClustering(n_clusters=3)
# 返回3个类别的中心坐标和各个元素的类别（295，296，294，因为前面的编号已经被用掉了）
centers, assignments = model.fit_transform(X)
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=assignments)
plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], c='b', s=180, marker='*')
plt.title("鸢尾花经过合并算法进行聚类")
plt.xticks([])
plt.yticks([])
plt.show()
