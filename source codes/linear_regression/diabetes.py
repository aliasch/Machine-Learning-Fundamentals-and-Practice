import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import linear_regression as lib
from sklearn.datasets import load_diabetes


def process_features(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])
    return X


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train = process_features(X_train)
X_test = process_features(X_test)
model = lib.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = model.mse(y_test, y_pred)
r2 = model.r2(y_test, y_pred)
print(f"mse = {mse}, r2 = {r2}")
