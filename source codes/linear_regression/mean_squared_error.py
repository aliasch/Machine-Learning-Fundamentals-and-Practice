import numpy as np
np.random.seed(1)
x = 10 * np.random.rand(10)
y = 2 * x + 5 + np.random.normal(0, 0.3, 10)
x_mean = np.mean(x)
y_mean = np.mean(y)
num = 0.0
d = 0.0
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
a = num / d
b = y_mean - a * x_mean
print(f"a={a}")
print(f"b={b}")
