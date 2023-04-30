import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from soft_svm_hinge import SoftSVMHinge

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = 2 * (iris["target"] == 2).astype(int).reshape(-1, 1) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

model = SoftSVMHinge(C=5.0)
w, b = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy= {accuracy}")

plt.axis([1, 7, 0, 3])
plt.scatter(X_train[:, 0][y_train[:, 0] == 1], X_train[:, 1][y_train[:, 0] == 1], marker='s', label='弗吉尼亚鸢尾花')
plt.scatter(X_train[:, 0][y_train[:, 0] == -1], X_train[:, 1][y_train[:, 0] == -1], marker='o', label='其他鸢尾花')
x0 = np.linspace(1, 7, 200)
# 分离直线、平面或超平面L: <w,x> + b = 0
# 即w_0 * x_0 + w_1 * x_1 + b = 0 => x1 = -w_0 / w_1 * x_0 - b / w_1
line = -w[0] / w[1] * x0 - b / w[1]
plt.plot(x0, line, color='black')
plt.legend()
plt.show()
