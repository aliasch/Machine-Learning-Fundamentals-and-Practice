import numpy as np
from matplotlib import pyplot as plt
# 若读者不是按照上面的目录创建，则应该根据自己创建的LinearRegression类所在的目录，适当的修改导入的路径。
import linear_regression as lib


def generate_samples(m):
    X = 10 * np.random.rand(m, 1)
    y = 2 * X + 5 + np.random.normal(0, 0.3, (m, 1))
    return X, y


def process_features(X):
    m, n = X.shape  # 获取训练样本的大小
    X = np.c_[np.ones((m, 1)), X]  # np.c_在这里的作用是在矩阵X的前面拼接一个mx1的全1向量，构成一个新的X
    return X


# 为了跟前面的例子保持一致，此处仍然使用同一个随机种子，产生10个样本。
np.random.seed(1)
X_train, y_train = generate_samples(10)
plt.scatter(X_train[:], y_train[:])
# 此处正是使用了前面所说的为了简化记号，增加了一个全1列，相当于把截距b也拿到w中
X_train = process_features(X_train)
# 调用模型，拟合出适当的参数w
model = lib.LinearRegression()
model.fit(X_train, y_train)
print(f"w_0 = {model.intercept_}, w_1 = {model.coef_}")
# 再次随机产生测试数据，并计算相应的均方误差和决定系数。
X_test, y_test = generate_samples(20)
X_test = process_features(X_test)
y_pred = model.predict(X_test)
mse = model.mse(y_test, y_pred)
r2 = model.r2(y_test, y_pred)
print(f"mse = {mse}, r2 = {r2}")
plt.plot(X_test[:, 1], y_pred)
plt.show()
