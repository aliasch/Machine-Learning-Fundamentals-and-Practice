import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph

# 生成瑞士卷数据集
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)
# 第1个维度的数值缩小为原来的0.5倍，是瑞士卷变薄
X[:, 1] *= .5
# 计算无结构的层级聚类
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
label = ward.labels_
# 绘图
fig = plt.figure()
ax = p3.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.view_init(7, -80)  # 调整图像视角
for l in np.unique(label):  # 过滤获取标签类型并排序，根据标签绘散点图，由标签值设置颜色
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2], color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
# 定义数据的结构A，最近邻域个数为10
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# 计算结构化的层级聚类
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward').fit(X)
label = ward.labels_
# 绘制结果
fig = plt.figure()
ax = p3.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2], color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.show()
