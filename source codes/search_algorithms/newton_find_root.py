import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def f(x):
    # return (x - 1) ** 2
    return x ** 2 - 4 * x + 3
# x^2 -4x + 3 =(x-1)(x-3)

def df(x):
    # return 2 * x - 2
    return 2 * x - 4


epsilon = 1e-6
x = 10.8
#x = 0.5
# x = 1.5
lstx = []
lsty = []
i = 1
while np.abs(f(x)) > epsilon:
    x = x - f(x) / df(x)
    lstx.append(x)
    lsty.append(f(x))
#     if i == 1 or i == 2:
#         plt.text(x + 0.02, f(x), f'第{i}步迭代')
#     i = i + 1
# plt.text(x, f(x) - 0.01, f'第{i}步迭代')
print(f"函数f(x)的根为{x}")
plt.scatter(lstx, lsty)
# x = np.linspace(0.7, 1.3, 100)
x = np.linspace(-1.7, 5.3, 100)
# plt.ylim([-0.02, 0.1])
plt.plot(x, f(x))
plt.plot(x, [0]*len(x))
plt.show()


#  x**2 - 4*x + 3