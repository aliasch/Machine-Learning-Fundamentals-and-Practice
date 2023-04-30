import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

# 创建50个可分离的点
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
# 拟合模型
model = SGDClassifier(loss="hinge", alpha=0.1, max_iter=1000)
model.fit(X, y)
# 绘制线，点以及距离超平面最近的向量
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)
X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = model.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
plt.contour(X1, X2, Z, levels, colors='k', linestyles=linestyles)
# levels = [-1.0, -0.5 , 0.0, 0.5, 1.0]
# linestyles = ['dashed', 'solid', 'dashed', 'solid', 'dashed']
# plt.contour(X1, X2, Z, levels, colors='k', linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,
            edgecolor='black', s=20)
plt.axis('tight')
plt.show()
