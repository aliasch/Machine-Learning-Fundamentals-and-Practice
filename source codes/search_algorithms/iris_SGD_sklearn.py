import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
iris = datasets.load_iris()
# 仅取出鸢尾花的前两个特征
X = iris.data[:, :2]
y = iris.target
colors = "bry"
# 打乱数据
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
# 标准化数据
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
h = .02  # 设置meshgrid中的步长
model = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)
# 创建绘制的mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 绘制决策边界，并设置mesh中每个点的颜色
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')
# 绘制训练集的数据点
for i, color in zip(model.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title("多分类SGD的决策表面")
plt.axis('tight')
# 绘制3个一对多的分类器
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = model.coef_
intercept = model.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)],
             ls="--", color=color)


for i, color in zip(model.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()
