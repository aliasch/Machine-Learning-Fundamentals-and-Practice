import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


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
    P = model.decision_function(xy).reshape(X.shape)  # 获得数据集X到分离超平面的距离
    # 绘制决策边界等高线
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['-.', '-', '--'])  # 生成等高线 --
    # ax.contourf(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])  # 生成等高线 --
    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=30, linewidth=10, marker='*')
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    plt.show()


def train_SVM():
    # n_samples=50 表示取50个点，centers=2表示将数据分为两类
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    # 线性核函数
    model = SVC(kernel='linear')
    model.fit(X, y)
    # print(model.coef_, model.intercept_)
    return model


model = train_SVM()
plot_SVC_decision_function(model)
