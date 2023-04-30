import matplotlib.pyplot as plt
import numpy as np

X = np.array([[0.2, 1, 2, 3, 4, 4, 5], [1.2, 1.5, 3, 3, 4.5, 5, 5]])
plt.scatter(X[0], X[1])
plt.xticks([0, 1, 2, 3, 4, 5, 6])
plt.yticks([0, 1, 2, 3, 4, 5, 6])
plt.xlabel('x', horizontalalignment='right', x=1.0)
plt.ylabel('y', horizontalalignment='right', y=1.0)

plt.scatter(X[0], X.shape[1] * [0])
plt.scatter(X.shape[1] * [0], X[1])

# (0.45, 0.95), (1, 1.5),  (2.25,2.75), (2.75, 3.25), (3.75,4.25),(4.25,4.75), (4.75,5.25)
XX = np.array([[0.45, 1, 2.25, 2.75, 4, 4.25, 4.75], [0.95, 1.5, 2.75, 3.25, 4.5, 4.75, 5.25]])
plt.scatter(XX[0], XX[1])
plt.plot(X[0], X[0] + 0.5)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_aspect(1)
plt.grid()
plt.show()
