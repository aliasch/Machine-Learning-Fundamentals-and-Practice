import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from decision_tree.decision_tree_classifier import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 用make_moons创建月亮形数据，make_circles创建环形数据，并将2组数据拼接到列表datasets中
datasets = [make_moons(n_samples=200, noise=0.2, random_state=1), make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)]
i = 1
for dataset in datasets:
    # 对X中的数据进行标准化处理，然后分训练集和测试集
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    # 找出数据集中两个特征的最大值和最小值，让最大值+0.5，最小值-0.5，创造一个比两个特征的区间本身更大一点的区间
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # 函数meshgrid用以生成网格数据，能够将两个一维数组生成两个二维矩阵。
    # 生成的网格数据，是用来绘制决策边界的，因为绘制决策边界的函数contourf要求输入的两个特征都必须是二维的
    x_axis, y_axis = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))
    ax = plt.subplot(len(datasets), 2, i)
    if i == 1:
        ax.set_title("月亮形数据")
    elif i == 3:
        ax.set_title("环形数据")
    # 先放训练集
    ax.scatter(X_train[:, 0], X_train[:, 1], edgecolors='k')
    # 放测试集
    ax.scatter(X_test[:, 0], X_test[:, 1], alpha=0.6)
    # 为图设置坐标轴的最大值和最小值，并设定没有坐标轴
    ax.set_xlim(x_axis.min(), x_axis.max())
    ax.set_ylim(y_axis.min(), y_axis.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i = i + 1
    ax = plt.subplot(len(datasets), 2, i)
    # 决策树的建模过程：实例化 → fit训练 → score接口得到预测的准确率
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    # 由于决策树在训练的时候导入的训练集X_train里面包含两个特征，所以我们在计算类概率的时候，也必须导入结构相同的数组，即是说，必须有两个特征
    # ravel()能够将一个多维数组转换成一维数组
    # 在这里，我们先将两个网格数据降维成一维数组，再将两个数组链接变成含有两个特征的数据，再代入决策树模型，
    # 生成的Z包含数据的索引和每个样本点对应的类概率，再切片，得到类概率值
    Z = model.predict_proba(np.c_[x_axis.ravel(), y_axis.ravel()])[:, 1]
    # 将返回的类概率作为数据，放到contourf里面绘制去绘制轮廓
    Z = Z.reshape(x_axis.shape)
    ax.contourf(x_axis, y_axis, Z, alpha=.8)
    # 将训练集、测试集放到图中去
    ax.scatter(X_train[:, 0], X_train[:, 1], edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], alpha=0.6)
    # 为图设置坐标轴的最大值和最小值
    ax.set_xlim(x_axis.min(), x_axis.max())
    ax.set_ylim(y_axis.min(), y_axis.max())
    # 设定坐标轴不显示标尺也不显示数字
    ax.set_xticks(())
    ax.set_yticks(())
    if i == 2:
        ax.set_title("决策树")
    elif i == 4:
        ax.set_title("决策树")
    # 展示性能评估的分数(Score)，分类的准确率
    ax.text(x_axis.max() - .3, y_axis.min() + .3, (f'{100 * score}%'), size=15, horizontalalignment='right')
    i = i + 1
plt.tight_layout()
plt.show()
