import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 创建数据集
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
# 拟合模型
max_depth = 4
regr_1 = DecisionTreeRegressor(max_depth=max_depth)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=10, random_state=rng)
regr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=30, random_state=rng)
# 绘制图像
plt.scatter(X, y, c="k", label="训练样本")
regr_l = [regr_1, regr_2, regr_3]
cs = ['r', 'g', 'b']
labels = ['n_estimators=1', 'n_estimators=10', 'n_estimators=300']
linestyles = ['-', '--', '-.']
for regr, c, label, linestyle in zip(regr_l, cs, labels, linestyles):
    regr.fit(X, y)
    y_ = regr.predict(X)
    plt.plot(X, y_, c, label=label, linewidth=2, linestyle=linestyle)
plt.xlabel("X")
plt.ylabel("y")
plt.title("提升的决策树回归")
plt.legend()
plt.show()
