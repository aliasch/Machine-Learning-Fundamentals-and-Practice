import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

train_scores, test_scores = [], []
ns = np.arange(1, 100, 5)
for n in ns:
    model = GradientBoostingClassifier(n_estimators=n, learning_rate=0.2, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))
plt.plot(ns, train_scores, label='训练样本分数')
plt.plot(ns, test_scores, label='测试样本分数')
print(
    f"训练集：{np.argmax(train_scores) * 5 + 1, np.max(train_scores)},测试集:{np.argmax(test_scores) * 5 + 1, np.max(test_scores)}")
plt.ylim([0.7, 1.05])
plt.xlabel("弱学习器个数")
plt.ylabel("分数")
plt.title('梯度提升分类器')
plt.legend(loc='lower right')
plt.show()
