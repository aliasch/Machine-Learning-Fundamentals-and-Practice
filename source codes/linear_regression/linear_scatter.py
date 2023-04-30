import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = 10 * np.random.rand(10)
y = 2 * x + 5 + np.random.normal(0, 0.3, 10)
plt.scatter(x, y)
# 导入sklearn，该包需要按照第1章所讲，先下载安装
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(x.reshape(-1, 1), y)
xfit = np.linspace(0, 10, 100)
yfit = model.predict(xfit.reshape(-1, 1))
plt.plot(xfit, yfit)
plt.show()
print(f"直线的系数：{model.coef_[0]}")
print(f"直线的截距：{model.intercept_}")
