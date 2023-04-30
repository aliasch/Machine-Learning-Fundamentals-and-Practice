import numpy as np


class KMeans:
    def __init__(self, n_clusters=1, max_iter=50, random_state=0):
        self.k = n_clusters
        self.N = max_iter
        np.random.seed(random_state)

    def assign_to_centers(self, centers, X):
        assignments = []  # assignments[i]记录样本X[i]被归入类的中心的编号，在循环中assignments.append
        # 一直往后追加，刚好跟i的下标一致。
        for i in range(len(X)):  # X[i]到各个中心centers[j]的距离distances，
            # 然后找出最小距离的编号np.argmin(distance)，将其加入到assignments数组
            distances = [np.linalg.norm(X[i] - centers[j], 2) for j in range(self.k)]
            assignments.append(np.argmin(distances))  # assignments[i]的值为X[i]所属的某一个类
        return assignments

    def adjust_centers(self, assignments, X):
        new_centers = []
        for j in range(self.k):
            # cluster_j是一个列表，保存了所有中心为类j的样本X[i]
            cluster_j = [X[i] for i in range(len(X)) if assignments[i] == j]
            # 求出新的类j的中心坐标
            new_centers.append(np.mean(cluster_j, axis=0))   # new_centers[j][i] 为类j中心坐标的第i个分量
        return new_centers

    def fit_transform(self, X):
        idx = np.random.randint(0, len(X), self.k)
        centers = [X[i] for i in idx]  # 初始化，生成k个中心坐标
        assignments = None
        for iter in range(self.N):
            assignments = self.assign_to_centers(centers, X)  # 样本归类
            centers = self.adjust_centers(assignments, X)  # 调整中心坐标
        return np.array(centers), np.array(assignments)
