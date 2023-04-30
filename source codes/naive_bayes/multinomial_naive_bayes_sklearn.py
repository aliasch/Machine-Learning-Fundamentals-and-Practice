# from sklearn.naive_bayes import MultinomialNB
# import numpy as np
#
# # 随机生成数据集
# X = np.random.randint(5, size=(6, 100))  # 生成形状为[6, 100]的0-4的整数
# y = np.array([1, 2, 3, 4, 5, 6])
# # 建立一个多项式朴素贝叶斯分类器
# model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# model.fit(X, y)
# # 由于概率永远是在[0,1]之间，mnb给出的是对数先验概率，因此返回的永远是负值，所以这里再取一个指数
# # 类先验概率=各类的个数/类的总个数
# print("类先验概率：", np.exp(model.class_log_prior_))
# print(f"每个标签类别下包含的样本数:{model.class_count_}")  # 因为每一个样本(每一行，即100个数据构成一个样本）都只出现了一次
# print(f"其中一个样本为:{X[2]}")
# print(f"预测的分类：{model.predict([X[2]])}")  # 输出3


from sklearn.naive_bayes import MultinomialNB
import numpy as np

# 随机生成数据集
np.random.seed(1)
X = np.random.randint(5, size=(6, 100))  # 生成形状为[6, 100]的0-4的整数
X = np.r_[X, X[2].reshape(1, -1)]
y = np.array([1, 2, 3, 4, 5, 6, 3])
# 建立一个多项式朴素贝叶斯分类器
model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model.fit(X, y)
# 由于概率永远是在[0,1]之间，mnb给出的是对数先验概率，因此返回的永远是负值，所以这里再取一个指数
# 类先验概率=各类的个数/类的总个数
print("类先验概率：", np.exp(model.class_log_prior_))
# 因为除了X[2]这个样本出现2次，其余每一个样本(每一行为一个样本，100个特征)都只出现了一次
print(f"每个标签类别下包含的样本数:{model.class_count_}")

print(f"其中一个样本为:{X[2]}")
print(f"预测的分类：{model.predict([X[1]])}")  # 输出3(对应的y[2]为3)
