import numpy as np


class Node:
    j = None
    theta = None
    p = None
    left = None
    right = None


class DecisionTreeBase:
    def __init__(self, max_depth, feature_sample_rate, regression_or_classification):
        self.max_depth = max_depth
        # 随机选取相应比例的特征进行遍历，随机森林才需要，决策树所有特征都要遍历到。
        self.feature_sample_rate = feature_sample_rate
        # 划分标准函数的指针（函数名），当传入的是方差，则实现决策树回归的cart算法；
        # 当传入的是熵，则实现决策树分类问题的cart算法
        self.regression_or_classification = regression_or_classification

    def split_data(self, j, theta, X, idx):
        idx1, idx2 = list(), list()
        for i in idx:  # idx是样本下标的列表
            value = X[i][j]  # 第i个样本的第j个特征的值
            if value <= theta:  # 若第i个样本的特征值X[i][j]<θ，则将该样本的下标i加入idx1列表中
                idx1.append(i)  # 所有第j个特征的值小于θ的样本都被分到idx1中
            else:
                idx2.append(i)
        return idx1, idx2

    def get_random_features(self, n):
        shuffled = np.random.permutation(n)
        size = int(self.feature_sample_rate * n)
        selected = shuffled[:size]
        return selected

    def find_best_split(self, X, y, idx):  # idx样本下标列表
        m, n = X.shape
        best_score = float("inf")
        best_j = -1
        best_theta = float("inf")
        best_idx1, best_idx2 = list(), list()
        selected_j = self.get_random_features(n)  # 特征下标集合
        for j in selected_j:
            thetas = set([x[j] for x in X])  # 被选中的特征中，样本特征的值构成的集合，消除重复
            for theta in thetas:
                idx1, idx2 = self.split_data(j, theta, X, idx)
                # 一个小集合idx_i如果为空，说明已经可以确定类型了，所以处理下一个特征
                if min(len(idx1), len(idx2)) == 0:
                    continue
                score1, score2 = self.regression_or_classification(y, idx1), self.regression_or_classification(y, idx2)
                w = 1.0 * len(idx1) / len(idx)  # w=avg(S_L) = \frac{|S_L|}{|S|}, 右子树的频率为1-w
                # 综合左右子树的分值（即：求出方差或交叉熵）以此作为当前划分的目标函数值
                score = w * score1 + (1 - w) * score2
                if score < best_score:  # 找出最好的，最小的方差或交叉熵对应的下标j和阈值θ
                    best_score = score
                    best_j = j
                    best_theta = theta
                    best_idx1 = idx1
                    best_idx2 = idx2
        return best_j, best_theta, best_idx1, best_idx2, best_score

    def generate_tree(self, X, y, idx, d):  # X样本，y标签，idx样本下标列表，d决策树的深度
        root = Node()
        root.p = np.average(y[idx], axis=0)  # 根据算法，算出标签的平均值
        if d == 0 or len(idx) < 2:
            return root
        # 调用方差或熵函数计算当前（不做划分时）的目标函数值（损失值）。
        current_score = self.regression_or_classification(y, idx)
        j, theta, idx1, idx2, score = self.find_best_split(X, y, idx)  # 采用贪心策略来寻找最优数据划分
        # 上一行的最佳划分的目标函数值为score，如果score小于current_score(方差或交叉熵，越小越好)，则数据划分可以带来益处，
        # 否则就不要划分了，直接返回root就好了。
        if score >= current_score:
            return root
        root.j = j
        root.theta = theta
        root.left = self.generate_tree(X, y, idx1, d - 1)  # 利用分出来的下标子集idx1，递归生成左子树
        root.right = self.generate_tree(X, y, idx2, d - 1)
        return root

    # CART算法的入口，在这里生成一棵深度不超过max_depth的决策树
    def fit(self, X, y):
        self.root = self.generate_tree(X, y, range(len(X)), self.max_depth)

    def get_prediction(self, root, x):  # root为根节点，x为单一条数据（样本）
        if root.left is None and root.right is None:
            return root.p
        value = x[root.j]  # root为一个node，具有j下标这个属性
        if value <= root.theta:  # 小于阈值，则从左子树进行预测
            return self.get_prediction(root.left, x)
        else:
            return self.get_prediction(root.right, x)

    def predict(self, X):
        y = list()
        for i in range(len(X)):
            y.append(self.get_prediction(self.root, X[i]))
        return np.array(y)
