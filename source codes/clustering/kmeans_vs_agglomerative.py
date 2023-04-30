import matplotlib.pyplot as plt
import numpy as np
from clustering.agglomerative_clustering import AgglomerativeClustering
from clustering.k_means import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_ball(x, radius, m):
    r = radius * np.random.rand(m)
    pi = 3.14
    theta = 2 * pi * np.random.rand(m)
    B = np.zeros((m, 2))
    for i in range(m):  # 圆的参数方程
        B[i][0] = x[0] + r[i] * np.cos(theta[i])
        B[i][1] = x[1] + r[i] * np.sin(theta[i])
    return B


B1 = generate_ball([0, 0], 1, 100)
B2 = generate_ball([0, 2], 1, 100)
B3 = generate_ball([5, 1], 0.5, 10)
X = np.concatenate((B1, B2, B3), axis=0)

kmeans = KMeans(n_clusters=2)
# kmeans = KMeans(n_clusters=3)
km_centers, km_assignments = np.array(kmeans.fit_transform(X), dtype=object)

agg = AgglomerativeClustering(n_clusters=2)
# agg = AgglomerativeClustering(n_clusters=3)
agg_centers, agg_assignments = agg.fit_transform(X)

plt.figure(figsize=(10, 3))
ax = plt.gca()
ax.set_aspect(1)
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='y')
plt.title('原始数据')
plt.text(1.5, 0, 'B1')
plt.text(1.5, 2, 'B2')
plt.text(4, 1, 'B3')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=km_assignments)
plt.scatter(km_centers[:, 0], km_centers[:, 1], c='r', marker='*', s=300)
plt.title('K均值聚类')
plt.text(1.5, 0, 'B1')
plt.text(1.5, 2, 'B2')
plt.text(4, 1, 'B3')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=agg_assignments)
plt.scatter(agg_centers[:, 0], agg_centers[:, 1], c='r', marker='*', s=300)
plt.title('合并聚类')
plt.text(1.5, 0, 'B1')
plt.text(1.5, 2, 'B2')
plt.text(4, 1, 'B3')
plt.show()
