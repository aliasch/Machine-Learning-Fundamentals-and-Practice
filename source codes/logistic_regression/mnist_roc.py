import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from logistic_regression.logistic_regression_gd import LogisticRegression


def threshold(t, proba):
    return (proba >= t).astype(int)


def roc(proba, y):
    fpr, tpr = [], []
    for i in range(100):
        z = threshold(0.01 * i, proba)
        tp = (y * z).sum()
        fp = ((1 - y) * z).sum()
        tn = ((1 - y) * (1 - z)).sum()
        fn = (y * (1 - z)).sum()
        fpr.append(1.0 * fp / (fp + tn))
        tpr.append(1.0 * tp / (tp + fn))
    return fpr, tpr


def process_features(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(1.0 * X)
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X

X, y = fetch_openml('mnist_784', data_home='~', version=1, return_X_y=True)
y = (np.array(y).astype(int) == 5.0).astype(int).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = LogisticRegression()
i = 1
for N in [2, 10, 20, 100]:
    model.fit(X_train, y_train, eta=.1, N=N)
    proba = model.predict_proba(X_test)
    fpr, tpr = roc(proba, y_test)
    plt.subplot(2, 2, i)
    plt.plot(fpr, tpr)
    i = i + 1
    plt.text(0.2, 0.8, f"AUC={auc(fpr, tpr):.2f}")
    plt.text(0.75, 0.2, f"N={N}")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
plt.tight_layout()
plt.show()
