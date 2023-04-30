import numpy as np
from sklearn.naive_bayes import BernoulliNB  # 伯努利朴素贝叶斯

x = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5]])
y = np.array([1, 1, 2])
model = BernoulliNB(alpha=2.0, binarize=3.0, fit_prior=True)
model.fit(x, y)
print(f"预测结果：{model.predict(np.array([[1, 2, 3, 4]]))}")  # 输出1
# class_log_prior_：类先验概率对数值
# print(f"类先验概率对数值：{model.class_log_prior_}")
# 类先验概率=各类的个数/类的总个数
print(f"类先验概率：{np.exp(model.class_log_prior_)}")
# feature_log_prob_：指定类的各特征概率(条件概率)对数值
# print(f"指定类的各特征概率(条件概率)对数值: {model.feature_log_prob_}")
# print(f"指定类的各特征概率(条件概率): {np.exp(model.feature_log_prob_)}")
# # 用伯努利分布公式计算,结果与上面的一样
# p_A_c1 = [(0 + 2) / (2 + 2 * 2) * 1,
#           (0 + 2) / (2 + 2 * 2) * 1,
#           (1 + 2) / (2 + 2 * 2) * 1,
#           (2 + 2) / (2 + 2 * 2) * 1]
# #          A   λ     B   λ
# # 上面A列表示：类别1中1的个数
# # 上面B列表示：类别1中样本数
# p_A_c2 = [(0 + 2) / (1 + 2 * 2) * 1,
#           (1 + 2) / (1 + 2 * 2) * 1,
#           (1 + 2) / (1 + 2 * 2) * 1,
#           (1 + 2) / (1 + 2 * 2) * 1]
# feature_prob = [p_A_c1, p_A_c2]
# print(f"公式计算得到的指定类的各特征概率：{np.array(feature_prob)}")
# # class_count_:按类别顺序输出其对应的样本数
# print(f"各类别的样本数: {model.class_count_}")
# # feature_count_:各类别各特征值之和，按类的顺序输出
# print(f"各类别各特征值之和: {model.feature_count_}")
