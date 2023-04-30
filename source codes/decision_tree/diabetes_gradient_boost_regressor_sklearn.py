import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载糖尿病数据集
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 5, 'learning_rate': 0.01, 'loss': 'squared_error'}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train, y_train)
mse = mean_squared_error(y_test, model.predict(X_test))
print(f"测试集上的最小均方误差MSE为{mse:.4f}")
# 绘制训练集偏差
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(model.staged_predict(X_test)):
    test_score[i] = model.loss_(y_test, y_pred)

plt.title('偏差')
plt.plot(np.arange(params['n_estimators']) + 1, model.train_score_, 'b-', label='训练集偏差')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='测试集偏差')
plt.legend(loc='upper right')
plt.xlabel('提升迭代次数')
plt.ylabel('偏差')
plt.tight_layout()
plt.show()
