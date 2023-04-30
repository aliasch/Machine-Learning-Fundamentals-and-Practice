import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置参数
n_repeat = 50  # 迭代次数
n_train = 50  # 训练集大小
n_test = 1000  # 测试集大小
noise = 0.1  # 噪声的标准差
np.random.seed(0)
# 设置评估器为决策树回归和Bagging回归, 高方差（例如决策树或KNN）效果更好
estimators = [("决策树回归", DecisionTreeRegressor()),
              ("Bagging回归", BaggingRegressor(DecisionTreeRegressor()))]
n_estimators = len(estimators)


# 生成数据
def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5  # [-5, 5)
    X = np.sort(X)
    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))
        for i in range(n_repeat):  # 迭代多次则生成多列数据
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y


X_train = []
y_train = []
for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)
X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)
plt.figure(figsize=(10, 8))
# 循环对比不同的评估器效果
for n, (name, estimator) in enumerate(estimators):
    # 计算预测值
    y_predict = np.zeros((n_test, n_repeat))
    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)
    # error（误差） = Bias^2（偏差） + Variance（方差） + Noise（噪声） 的均方误差分解
    y_error = np.zeros(n_test)
    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2
    y_error /= (n_repeat * n_repeat)
    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)
    print(
        f"{name}: {np.mean(y_error):.4f} (error) = {np.mean(y_bias):.4f} (bias^2) + {np.mean(y_var):.4f} (var) + {np.mean(y_noise):.4f} (noise)")
    # 绘制图像
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)=e^{-x^{2}}+1.5e^{-(x-2)^{2}}$")
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")  # Learning Sample
    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label=r"$\^y(x)$")
        else:
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)
    plt.plot(X_test, np.mean(y_predict, axis=1), "c",
             label=r"$\mathbb{E}_{LS} \^y(x)$")
    plt.xlim([-5, 5])
    plt.title(name)
    if n == n_estimators - 1:
        plt.legend(loc=(1.1, .5))
    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$")
    plt.plot(X_test, y_var, "g", label="$variance(x)$")
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")
    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])
    if n == n_estimators - 1:
        plt.legend(loc=(1.1, .5))
plt.subplots_adjust(right=.75)
plt.show()

#  https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py