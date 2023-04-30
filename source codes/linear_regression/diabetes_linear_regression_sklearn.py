import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载糖尿病数据集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
# 仅取出BMI(body mass index)身体质量指数1个特征
X = X[:, 2].reshape(-1, 1)
# 将数据分成训练集和测试集以及它们对应的值
X_train, X_test, y_train, y_test = X[:-20], X[-20:], y[:-20], y[-20:]
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'系数coe：{model.coef_}')
print(f'均方误差mse：{mean_squared_error(y_test, y_pred):.2f}')
print(f'决定系数R^2：{r2_score(y_test, y_pred):.2f}')
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('糖尿病数据集的线性回归')
plt.xticks(())
plt.yticks(())
plt.show()
