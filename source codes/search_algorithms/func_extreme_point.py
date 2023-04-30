import numpy as np


def f(x):
    return x ** 2 + 3 * x + 4


def df(x):
    return 2 * x + 3


def ddf(x):
    return 2


x = 0
epsilon = 1e-6
while np.abs(df(x)) > epsilon:
    x = x - df(x) / ddf(x)
print(f"函数f(x)=x^2 + 3x + 4在{x}点处取得极小值为: {f(x)}")
