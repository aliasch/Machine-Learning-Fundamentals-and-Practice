import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def plot_SVC_decision_function(model, ax=None, plot_support=True):
    """绘制2维支持向量机分类器的决策函数"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 创建评估模型的网格数据
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    # 生成网格点和坐标矩阵
    Y, X = np.meshgrid(y, x)
    # 拼接数组
    xy = np.c_[X.ravel(), Y.ravel()]
    P = model.decision_function(xy).reshape(X.shape)
    # 绘制决策边界等高线
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])  # 生成等高线 --
    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=5, linewidth=10, marker='*')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#
# X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.6)
# fig, ax = plt.subplots(1, 2)
# for axi, C in zip(ax, [10.0, 0.1]):
#     model = SVC(kernel='linear', C=C)
#     model.fit(X, y)
#     axi.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='autumn')
#     plot_SVC_decision_function(model, axi)
#     axi.set_title(f'C={C}', size=14)


X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.6)
fig, ax = plt.subplots(1, 3)
for axi, gamma in zip(ax, [10.0, 1.0, 0.1]):
    model = SVC(kernel='rbf', gamma=gamma)
    model.fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_SVC_decision_function(model, axi)
    axi.set_title(f'gamma={gamma}', size=14)

plt.tight_layout()
plt.show()
