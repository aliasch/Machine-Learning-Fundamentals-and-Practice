import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 产生样本数据并标准化
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X)
# 计算DBSCAN，设置核心样本掩码
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # 产生一个跟db.labels_形状相同的全0数组
core_samples_mask[db.core_sample_indices_] = True  # 将核心样本点设置为True，
labels = db.labels_
# 因为噪声的标签为-1，如果存在噪声，聚类的类别要减去1，忽略噪声的类别
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
# 打印各种指标信息
# print(f'估计类别数为:{n_clusters_}')
# print(f'估计噪声数：{n_noise_}')
# print(f"同种类:{metrics.homogeneity_score(labels_true, labels):.3f}")
# print(f"完整性：{metrics.completeness_score(labels_true, labels):.3f}")
# print(f"V-measure:{metrics.v_measure_score(labels_true, labels):.3f}")
# print(f"调整兰德系数:{metrics.adjusted_rand_score(labels_true, labels):.3f}")
# print(f"调整的互信息：{metrics.adjusted_mutual_info_score(labels_true, labels):.3f}")
# print(f"轮廓系数:{metrics.silhouette_score(X, labels):.3f}")
# 移除黑色，将黑色用于噪声。
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, color in zip(unique_labels, colors):
    if k == -1:  # k==-1的标签就是噪声，# 黑色用于噪声,rgba(0,0,0,1) 则表示完全不透明的黑色,alpha为透明度
        color = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    # 使用掩码，按位与，同时是核心样本，又是黑色的样本,将其画大一点，填充颜色由tuple(color)指定，边缘颜色为黑色
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]  # 不是核心样本,画小一点
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='y', markersize=6)
plt.title(f'估计的类别数量:{n_clusters_}')
plt.show()
