import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
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


def train_svm_circles():
    # 二维圆形数据 factor 内外圆比例（0, 1）
    X, y = make_circles(100, factor=0.4, noise=0.1)
    # model = SVC(kernel='linear')
    model=SVC(kernel='rbf')
    model.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_SVC_decision_function(model, plot_support=True)
    # plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=5, linewidth=10, marker='*')
    return X, y


def plot_3D(X, y, elev=30, azim=30):
    # 加入了新的维度 z
    z = np.exp(-(X ** 2).sum(axis=1))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], z, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)  # 调整视角
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


X, y = train_svm_circles()
plot_3D(elev=30, azim=30, X=X, y=y)

plt.show()
