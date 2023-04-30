import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from softmax_regression_gd import SoftmaxRegression
from metrics import accuracy_score


def one_hot_encoding(y):
    m = len(y)
    k = np.max(y) + 1
    labels = np.zeros([m, k])
    for i in range(m):
        labels[i, y[i]] = 1
    return labels


def process_features(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(1.0 * X)
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


iris = datasets.load_iris()
X = iris["data"]
c = iris["target"]
y = one_hot_encoding(c)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train = process_features(X_train)
X_test = process_features(X_test)
model = SoftmaxRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
y_pred = one_hot_encoding(y_pred)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy = {accuracy}")
