import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 固定种子，生成样本数据
np.random.seed(0)
batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
# 用kmeans算法计算X聚类的结果
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
# t0 = time.time()
k_means.fit(X)
# t_batch = time.time() - t0
# 用MiniBatchkmeans算法计算X聚类的结果
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size, n_init=10, max_no_improvement=10,
                      verbose=0)
# t0 = time.time()
mbk.fit(X)
# t_mini_batch = time.time() - t0
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
# 为了使得同一个类别的数据具有相同的颜色，做下面的操作
# 计算小批量kmeans算法的3个质心跟kmeans算法的质心，两两之间，哪些才是距离最近的,order=[1 2 0]表示
# kmeans的第0个跟minikmeans的第1个距离最近，第1个跟第2个距离最近，第2个跟第0个最近
k_means_cluster_centers = k_means.cluster_centers_
order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
# 按order排序得到的mkb_means_cluster_centers，它的质心跟k_means_clusters_centers相同下标取出的质心就是距离最小的
mbk_means_cluster_centers = mbk.cluster_centers_[order]
# 计算各个样本跟质心的最小距离，从而得到各个点的类别
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
# KMeans# 先利用  k_means_labels == k取定样本中标签为k的有哪一些（结果为True的就是）
# 将结果保存在my_members中，这些点就应该具有相同的颜色，就可以绘图了，k循环3次，就得到了3种类别样本的散点图
ax = fig.add_subplot(1, 3, 1)
for k, color in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]  # 类别中心，即质心
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=color, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)
ax.set_title('k均值算法')
ax.set_xticks(())
ax.set_yticks(())
# print(f'训练时间：{t_batch:.2f}秒\n惯量：{k_means.inertia_}')
# MiniBatchKMeans 下面的处理跟Kmeans完全相同
ax = fig.add_subplot(1, 3, 2)
for k, color in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=color, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)
ax.set_title('小批量kmeans算法')
ax.set_xticks(())
ax.set_yticks(())
# print(f'训练时间: {t_mini_batch:.2f}秒,\n惯量：{mbk.inertia_}')

# 初始化不同样本点为全0（数组长度跟mbk_means_labels一样，所以利用它来赋值就好了）
# 因为mbk_means_labels的标签值为0，1，2，不可能是4，所以different全部被初始化为false
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)
# 通过循环，分别获得跟相同类别（0,1,2），在两种不同聚类算法下的不同样本，将对比结果存放在different中
for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == k))
# 逻辑取反，得到的就是相同的点，从而下面可以将相同和不同类别的点，以不同的颜色绘制出来
identic = np.logical_not(different)
ax.plot(X[identic, 0], X[identic, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
ax.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='m', marker='.')
ax.set_title('不同处的地方为深颜色的点')
ax.set_xticks(())
ax.set_yticks(())
plt.show()
