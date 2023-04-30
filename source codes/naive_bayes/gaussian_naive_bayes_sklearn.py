from sklearn.naive_bayes import GaussianNB  # 高斯分布，假定特征服从正态分布的
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = GaussianNB()
model.fit(X_train, y_train)
predict_class = model.predict(X_test)
print(model.predict([[5.0, 3.5, 1.4, 0.2]]))
print(model.predict(np.array([7, 2.2, 4.7, 1.4]).reshape(1, -1))[0])
print(model.predict(np.array([5.8, 3.6, 5.1, 1.8]).reshape(1, -1)))
print(f"测试集准确率为：{accuracy_score(y_test, predict_class)}")
