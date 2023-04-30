import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import linear_regression as lib


def generate_samples(m):
    X = 4 * np.random.rand(m, 1) - 2
    y = X ** 3 + 2 * X ** 2 - 3 * X + 5 + np.random.normal(0, 0.2, (m, 1))
    return X, y


np.random.seed(1)
X, y = generate_samples(100)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = lib.LinearRegression()
model.fit(X_poly, y)
print(f"w_0 = {model.intercept_}, coef_ = {model.coef_}")
# 再次随机产生测试数据，并计算多项式回归相应的均方误差和决定系数。
X_test, y_test = generate_samples(20)
X_test = poly.fit_transform(X_test)  # 测试时，使用上面得到的多项式poly
y_pred = model.predict(X_test)
mse = model.mse(y_test, y_pred)
r2 = model.r2(y_test, y_pred)
print(f"mse = {mse}, r2 = {r2}")
plt.figure(0)
plt.scatter(X, y)
plt.figure(1)
plt.scatter(X, y)
W = np.linspace(-2, 2, 100).reshape(100, 1)  # 拟合曲线，点集W拟合出u
W_poly = poly.fit_transform(W)
u = model.predict(W_poly)
plt.plot(W, u)
plt.show()
