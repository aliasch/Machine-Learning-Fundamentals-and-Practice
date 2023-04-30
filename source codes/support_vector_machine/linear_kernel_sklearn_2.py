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
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['-.', '-', '--'])  # 生成等高线 --
    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=30, linewidth=10, marker='*')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.5)
    X, y = X[:N], y[:N]
    model = SVC(kernel='linear')
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_SVC_decision_function(model, ax)


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2)
    for axi, N in zip(ax, [60, 120]):
        plot_svm(N, axi)
        axi.set_title(f'N = {N}')
    plt.tight_layout()
    plt.show()
