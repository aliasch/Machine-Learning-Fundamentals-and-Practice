import numpy as np
import matplotlib.pyplot as plt

# 在创建任意一个普通坐标轴的过程中，加入projection='3d'关键字，就可创建3为坐标轴
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(n=50):
    x = np.random.randn(n)
    y = np.random.randn(n)
    x1, y1 = x[np.abs(x - y) > 1], y[np.abs(x - y) > 1]
    x2, y2 = x[np.abs(x - y) < 1], y[np.abs(x - y) < 1]
    return x1, y1, x2, y2


np.random.seed(10)
x1, y1, x2, y2 = generate_data()
plt.scatter(x1, y1, marker='*', label='次品')
plt.scatter(x2, y2, marker='x', label='正品')
u1 = np.linspace(-3, 3, 50)
v1 = u1 + 1
plt.plot(u1, v1)
u2 = np.linspace(-3, 3, 50)
v2 = u1 - 1
plt.plot(u2, v2)
plt.xticks([-3, 3])
plt.yticks([-3, 3])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()
ax = plt.axes(projection='3d')
z1 = np.abs(x1 - y1)
z2 = np.abs(x2 - y2)
_ = ax.scatter3D(x1, y1, z1, c=z1, marker='x', cmap='Blues')
_ = ax.scatter3D(x2, y2, z2, c=z2, marker='o', cmap='Greens')

w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)
Z2 = np.ones(W1.shape)
_ = ax.plot_surface(W1, W2, Z2, alpha=0.1)
ax.view_init(elev=2, azim=25)  # 改变绘制图像的视角
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 3)  # 限制坐标的尺寸
plt.show()
