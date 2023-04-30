from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression_gd import LogisticRegression
import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 0).astype(int).reshape(-1, 1)
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
print(f"cross entropy = {entropy}")
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"accuracy = {accuracy}")
print("山鸢尾花的标签为1，非山鸢尾花的标签为0")
sample = X[0].reshape(1, -1)
print(f"X[0]的真实标签为{y[0, 0]}")
sample = process_features(sample)
result = model.predict(sample)
print(f"样本被预测为{result[0, 0]}")
proba = model.predict_proba(sample)
print(f"预测结果为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
sample = X[140].reshape(1, -1)
print(f"X[140]的真实标签为{y[140, 0]}")
sample = process_features(sample)
result = model.predict(sample)
print(f"样本被预测为{result[0, 0]}")
proba = model.predict_proba(sample)
print(f"预测结果为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
