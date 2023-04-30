import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()

X = iris.data
y = iris.target
# target_names = iris.target_names
target_names = ['山鸢尾花', '变色鸢尾花', '弗吉尼亚鸢尾花']
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('鸢尾花数据集的PCA降维')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()
