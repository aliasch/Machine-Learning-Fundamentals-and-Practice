import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn import datasets
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering

X, y = datasets.load_digits(return_X_y=True)
n_samples, n_features = X.shape
np.random.seed(0)


def nudge_images(X, y):
    # shift()  # 使用请求顺序的样条插值来移动数组。 根据给定的模式填充输入边界外的点。
    # 第一个参数是输入，数组类型    # 第二个参数是偏移量（[行，列]）   # 第三个参数是填充数
    shift = lambda x: ndimage.shift(x.reshape((8, 8)), .3 * np.random.normal(size=2),
                                    mode='constant', ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


# 可视化聚类
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)
    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# 数字数据集的2维嵌入（使用SpectralEmbedding进行非线性降维）
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
# 使用不同的linkage链接度量标准进行聚类
for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    clustering.fit(X_red)
    plot_clustering(X_red, clustering.labels_, f"{linkage}")
plt.show()
