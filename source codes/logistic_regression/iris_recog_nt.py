import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression_nt import LogisticRegression
import metrics


def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 1).astype(int).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = LogisticRegression()
model.fit(X_train, y_train, N=5000)
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

print("变色鸢尾花的标签为1，非变色鸢尾花的标签为0")
sample = X[0].reshape(1, -1)
print(f"X[0]的真实标签为{y[0, 0]}")
sample = process_features(sample)
result = model.predict(sample)
print(f"样本被预测为{result[0, 0]}")
proba = model.predict_proba(sample)
print(f"预测结果为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
sample = X[60].reshape(1, -1)
print(f"X[60]的真实标签为{y[60, 0]}")
sample = process_features(sample)
result = model.predict(sample)
print(f"样本被预测为{result[0, 0]}")
proba = model.predict_proba(sample)
print(f"预测结果为0和1的概率分别为：{1 - proba[0, 0], proba[0, 0]}")
