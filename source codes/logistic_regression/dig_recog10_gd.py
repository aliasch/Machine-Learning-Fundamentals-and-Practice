import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from logistic_regression_gd import LogisticRegression
import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


# 加载digits数据集
digits = load_digits()
data = digits.data
target = digits.target
# 取出0和1的数据
zeros = data[target == 0]
ones = data[target == 1]
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
entropy = metrics.cross_entropy(y_test, proba)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"自己实现的cross entropy = {entropy}")
print(f"sklearn实现的cross entropy = {log_loss(y_test, proba)}")
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"accuracy = {accuracy}")
# 拿出一个其他数据进行识别例如0和5
zero = data[target == 0][0].reshape(1, -1)
plt.imshow(zero.reshape(8, 8))
plt.show()
zero = process_features(zero)
result = model.predict(zero)
print(f"0被预测为{result[0, 0]}")
proba = model.predict_proba(zero)
print(f"预测结果为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
# 拿出一个其他数据进行识别例如5
five = data[target == 5][0].reshape(1, -1)
plt.imshow(five.reshape(8, 8))
plt.show()
five = process_features(five)
result = model.predict(five)
print(f"5被预测为{result[0, 0]}")
proba = model.predict_proba(five)
print(f"预测结果为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
