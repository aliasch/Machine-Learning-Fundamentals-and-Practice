import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 加载digits数据集
digits = load_digits()
data = digits.data
target = digits.target
# 打印数据集大小
# print(data.shape)
# print(target.shape)
# # 先展示数字0-9的例子
for i in range(0, 10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(data[target == i][0].reshape(8, 8))
plt.show()
# 取出0和1的数据
zeros = data[target == 0]
ones = data[target == 1]
# print(zeros.shape) #可以查看0和1的个数
# print(ones.shape)
# 将0和1数据按行拼接起来
X = np.vstack([zeros, ones])
# 生成0,1对应的标签
y = np.array([0] * zeros.shape[0] + [1] * ones.shape[0])
# 对数据进行训练样本和测试样本的切分，然后调用Logistic回归模型，进行训练，预测
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(y_pred)
score = model.score(X_test, y_test)
print(f"测试的结果准确率:{score}")
# 拿出一个其他数据进行识别例如0
zero = data[target == 0][0]
plt.imshow(zero.reshape(8, 8))
plt.show()
result = model.predict(zero.reshape(1, -1))
print(f"0被预测为{result[0]}")
proba = model.predict_proba(zero.reshape(1, -1))
print(f"预测结果的概率为0和1的概率分别为：{proba}")
# 拿出一个其他数据进行识别例如5
five = data[target == 5][0]
plt.imshow(five.reshape(8, 8))
plt.show()
result = model.predict(five.reshape(1, -1))
print(f"5被预测为{result[0]}")
proba = model.predict_proba(five.reshape(1, -1))
print(f"预测结果的概率为0和1的概率分别为：{proba}")
