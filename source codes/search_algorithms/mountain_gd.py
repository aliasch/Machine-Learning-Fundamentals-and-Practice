import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.seterr(divide='ignore', invalid='ignore')
def f(x):
    return (np.cos(1/x)) ** 2
def df(x):
    return np.sin(2 / x) / (x ** 2)
X, y = [], []
eta, epsilon = 0.001, 0.0001 # 学习率eta和终止条件epsilon
x = 0.3
# x = 0.4
i = 1
plt.text(x, f(x) + 0.1, '起始点')
while abs(df(x)) > epsilon:
    X.append(x)
    y.append(f(x))
    if i % 4 == 0:
    # if i == 20 or i == 100:
         plt.text(x-0.05, f(x), f'第{i}步')
    x = x - eta * df(x)
    i = i + 1
print(f"函数经过{i}次迭代，极值点为：(x, f(x)={x, f(x)}")
plt.text(x, f(x) - 0.1, '终止点')
W = np.linspace(0, 1, 500).reshape(500, 1)
U = f(W)
plt.plot(W, U)
plt.scatter(X, y, s=15)
plt.xlim([0.1, 0.7])
plt.ylim([-0.5, 1.5])
plt.show()
