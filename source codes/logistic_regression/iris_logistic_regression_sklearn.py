import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
iris = load_iris()
X, y = iris.data[:, :2], iris.target
nums = [1, 2]
multi_classes = ['multinomial', 'ovr']
for num, multi_class in zip(nums, multi_classes):
    plt.subplot(1, 2, num)
    model = LogisticRegression(solver='sag', max_iter=5000, random_state=0, multi_class=multi_class, C=1e2)
    model.fit(X, y)
    # 打印训练集分数
    print(f"训练分数：{model.score(X, y):.3f}, {multi_class}")
    # 构建网格，产生一些介于最值之间的等差数据
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 预测网格内的数据所属类型，并可视化，绘制决策边界，并对每一种类型的点分配一种颜色
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.title(f"{multi_class}")
    plt.xlabel('花瓣长度')
    plt.ylabel('花瓣宽度')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    # 绘制训练数据的散点图
    colors = "bry"
    for i, color in zip(model.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired, edgecolor='black', s=20)
    # 绘制 3 个一对多的分类器超平面
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

plt.show()
