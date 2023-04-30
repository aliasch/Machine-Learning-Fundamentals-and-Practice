import numpy as np
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
from decision_tree_regressor import DecisionTreeRegressor
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(1)
X = np.linspace(0, 3 * np.pi, 50).reshape(-1, 1)
y = np.random.normal(np.cos(X[:, 0]), 0.1)
plt.subplot(3, 2, 1)
plt.scatter(X[:, 0], y, s=10)
plt.title('原始数据' )
for i in range(1, 6):
    plt.subplot(3, 2, i+1)
    plt.scatter(X[:, 0], y, s=10)
    model = DecisionTreeRegressor(max_depth=i)
    model.fit(X, y)
    y_pred = model.predict(X)
    X_, y_pred = zip(*sorted(zip(X[:, 0], y_pred)))
    plt.step(X_, y_pred, c='red')
    plt.title(f"决策树拟合深度{i}")
plt.tight_layout()
plt.show()