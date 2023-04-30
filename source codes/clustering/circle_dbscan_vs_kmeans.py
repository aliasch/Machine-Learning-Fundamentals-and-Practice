import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import clustering.dbscan as db
import clustering.k_means as km

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# np.random.seed(0)
# X_circles, y_circles = make_circles(n_samples=400, factor=.3, noise=.05)
X_circles, y_circles = make_circles(n_samples=100, factor=.5, noise=.05)
# X_moons, y_moons = make_moons(n_samples=400, noise=.05)
X_moons, y_moons = make_moons(n_samples=100, noise=.105)
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_aspect(1)
eps = 0.3
# eps = 0.5
min_sample=5
plt.subplot(2, 3, 1)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles)
plt.title('原始数据')

dbscan = db.DBSCAN(eps=eps, min_sample=min_sample)
db_circles_assignments = dbscan.fit_transform(X_circles)
plt.subplot(2, 3, 2)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=db_circles_assignments)
plt.title('DBSCAN聚类')

kmeans = km.KMeans(n_clusters=2)
km_centers, km_circles_assignments = kmeans.fit_transform(X_circles)
plt.subplot(2, 3, 3)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=km_circles_assignments)
plt.scatter(km_centers[:, 0], km_centers[:, 1], c='r', marker='*', s=300)
plt.title('k均值聚类')

plt.subplot(2, 3, 4)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)

db_moons_assignments = dbscan.fit_transform(X_moons)
plt.subplot(2, 3, 5)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=db_moons_assignments)

km_centers, km_moons_assignments = kmeans.fit_transform(X_moons)
plt.subplot(2, 3, 6)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=km_moons_assignments)
plt.scatter(km_centers[:, 0], km_centers[:, 1], c='r', marker='*', s=300)
plt.show()
