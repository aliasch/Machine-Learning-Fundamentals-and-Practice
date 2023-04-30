import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
datasets = [make_moons(n_samples=200, noise=0.2, random_state=1), make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)]
i = 1
for dataset in datasets:
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x_axis, y_axis = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))
    ax = plt.subplot(len(datasets), 2, i)
    if i == 1:
        ax.set_title("月亮形数据")
    elif i == 3:
        ax.set_title("环形数据")
    ax.scatter(X_train[:, 0], X_train[:, 1], edgecolors='k', marker='*')
    ax.scatter(X_test[:, 0], X_test[:, 1], alpha=0.6)
    ax.set_xlim(x_axis.min(), x_axis.max())
    ax.set_ylim(y_axis.min(), y_axis.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i = i + 1
    ax = plt.subplot(len(datasets), 2, i)
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    Z = model.predict_proba(np.c_[x_axis.ravel(), y_axis.ravel()])
    Z = Z[:, 1]
    Z = Z.reshape(x_axis.shape)
    ax.contourf(x_axis, y_axis, Z, alpha=.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], alpha=0.6)
    ax.set_xlim(x_axis.min(), x_axis.max())
    ax.set_ylim(y_axis.min(), y_axis.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if i == 2:
        ax.set_title("决策树")
    elif i == 4:
        ax.set_title("决策树")
    ax.text(x_axis.max() - .3, y_axis.min() + .3, (f'{100 * score}%'), size=15, horizontalalignment='right')
    i = i + 1
plt.tight_layout()
plt.show()
