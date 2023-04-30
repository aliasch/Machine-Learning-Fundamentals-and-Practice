import matplotlib.pyplot as plt
import numpy as np
from logistic_regression_gd import LogisticRegression
import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.datasets import fetch_openml

np.seterr(divide='ignore')  # RuntimeWarning: divide by zero encountered in log
np.seterr(invalid='ignore')  # RuntimeWarning: invalid value encountered in multiply


def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


mnist = fetch_openml('mnist_784', data_home='./mnist_data')  # 可以不指定data_home
X, y = mnist.data, mnist.target  #
X = np.array(X)
y = np.array(y)
zeros = X[y == '0']
ones = X[y == '1']
# 将0和1数据按行拼接起来
X = np.vstack([zeros, ones])
# 生成0,1对应的标签
y = np.array([0] * zeros.shape[0] + [1] * ones.shape[0])
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = LogisticRegression()
model.fit(X_train, y_train, eta=0.1, N=5000)
proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
# entropy1 = metrics.cross_entropy(y_test, proba)
entropy2 =log_loss(y_test, proba)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)


# print(f"自己实现的cross entropy = {entropy1}")
print(f"cross entropy = {entropy2}")
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"accuracy = {accuracy}")

# 拿出一个其他数据进行识别例如0和1
digit = X_test[10]
# plt.imshow(digit.reshape(28, 28))
# plt.show()
# digit = process_features(digit.reshape(1, -1))
result = model.predict(digit)
print(f"X_test[0]被预测为{result[0, 0]}")
proba = model.predict_proba(digit)
print(f"预测结果的概率为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
# 拿出一个其他数据进行识别例如0和1
digit = X[11000]
plt.imshow(digit.reshape(28, 28))
plt.show()
digit = process_features(digit.reshape(1, -1))
result = model.predict(digit)
print(f"X[11000]被预测为{result[0, 0]}")
proba = model.predict_proba(digit)
print(f"预测结果的概率为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
