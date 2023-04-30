from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 产生样本
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
# 计算近邻传播
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_  # 聚类中心下标
labels = af.labels_  # 聚类结果的类别
n_clusters_ = len(cluster_centers_indices)
from itertools import cycle

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, color in zip(range(n_clusters_), colors):
    class_members = labels == k  # 由labels==k确定同类样本点
    cluster_center = X[cluster_centers_indices[k]]  # 聚类中心的样本
    plt.plot(X[class_members, 0], X[class_members, 1], color + '.')  # 使用相同颜色绘制同类的点
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14)  # 绘制聚类中心
    for x in X[class_members]:  # 将聚类中心与相同分类的点用直线连接起来
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color)
plt.title(f'预测的聚类数:{n_clusters_}')
plt.show()
