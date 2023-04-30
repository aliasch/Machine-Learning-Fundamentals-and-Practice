import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1, 2, 1)
x1 = np.arange(-4, -1)
x2 = np.arange(-1, 4)
x3 = np.arange(4, 6)
y1 = np.zeros(len(x1))
y2 = np.zeros(len(x2))
y3 = np.zeros(len(x3))
plt.scatter(x1, y1, marker='x', c='b')
plt.scatter(x2, y2, marker='+', c='g')
plt.scatter(x3, y3, marker='x', c='b')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.title('数据在一维空间，线性不可分')
plt.subplot(1, 2, 2)
yy1 = x1 ** 2
yy2 = x2 ** 2
yy3 = x3 ** 2
plt.scatter(x1, yy1, marker='x', c='b')
plt.scatter(x2, yy2, marker='+', c='g')
plt.scatter(x3, yy3, marker='x', c='b')
u = np.linspace(-3, 7, 10)
v = 2 * u + 5.5
plt.plot(u, v)
plt.xlabel('x')
plt.ylabel('$x^{2}$')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.title('将数据投影到二维空间，线性可分')
plt.tight_layout()
plt.show()
