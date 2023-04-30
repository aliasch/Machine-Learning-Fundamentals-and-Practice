import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载手写数字的数据集
data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
# 为了便于可视化，将手写数字8x8=64个特征使用PCA算法降为2维数据。
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)
# 设置网格图的步长大小，步长越小图的质量越高
h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
# 绘制决策边界，给每一个边界分配一种颜色
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 预测网格中每个点的标签
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# 将结果用带颜色的图像（色块）展示出来，
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect="auto", origin="lower")
# 绘制经过降维后得各个数据点的散点图
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, s=1)
# 以白色X的形状，绘制类别中心点
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
plt.title("经PCA降维后的手写数字集用k均值聚类")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
