import numpy as np
from decision_tree.decision_tree_base import DecisionTreeBase


def get_var(y, idx):
    # 获得一个所有元素值为标签平均值的向量，长度为len(idx)
    y_avg = np.average(y[idx]) * np.ones(len(idx))
    return np.linalg.norm(y_avg - y[idx], 2) ** 2 / len(idx)  # 求出方差，结果为一个实数


class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, max_depth=0, feature_sample_rate=1.0):
        super().__init__(max_depth=max_depth, feature_sample_rate=feature_sample_rate,
                         regression_or_classification=get_var)

    def score(self, X, y_true):   # 这是决定系数R^{2}
        y_pred = self.predict(X)
        numerator = (y_true - y_pred) ** 2
        denominator = (y_true - np.average(y_true)) ** 2
        return 1 - numerator.sum() / denominator.sum()
