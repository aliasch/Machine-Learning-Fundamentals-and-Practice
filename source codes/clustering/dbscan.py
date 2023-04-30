import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_sample=5):
        self.assignments = None
        self.eps = eps
        self.min_sample = min_sample

    def get_neighbors(self, X, i):  # 获取x[i]的邻域
        m = len(X)
        distances = [np.linalg.norm(X[i] - X[j], 2) for j in range(m)]  # 样本i到各个样本j的距离
        neighbors_i = [j for j in range(m) if distances[j] < self.eps]
        return neighbors_i   # 返回X[i]的邻域元素的下标列表

    def grow_cluster(self, X, i, neighbors_i, id):  # 当前的类为id
        self.assignments[i] = id  # 将样本i归入当前类id中
        Q = neighbors_i  # neighbors_i是前面计算出来的样本i的邻域样本下标列表，将其加入队列Q中
        t = 0  # t是队首元素在Q中的下标
        while t < len(Q):  # 这个len(Q)在循环过程中，会跟着下面neighbors_j的加入而变化
            j = Q[t]  # 出队（取出队列中的元素）
            t += 1
            if self.assignments[j] == 0:  # 若出队元素j尚未归入某一类，则将其归入当前类id中
                self.assignments[j] = id
                neighbors_j = self.get_neighbors(X, j)  # 计算获取样本j的邻域样本
                if len(neighbors_j) > self.min_sample:  # 若j的邻域样本neighbors_j足够稠密，则将这些样本也加入队列
                    Q += neighbors_j

    def fit_transform(self, X):
        self.assignments = np.zeros(len(X))
        id = 1
        for i in range(len(X)):
            if self.assignments[i] != 0:  # 若样本i已分好类了，就跳过
                continue
            neighbors_i = self.get_neighbors(X, i)  # 获取样本i的邻域
            if len(neighbors_i) > self.min_sample:  # 若样本i的邻域是稠密邻域，则添加新的类为id
                self.grow_cluster(X, i, neighbors_i, id)
                id += 1
        return self.assignments  # 返回各个样本的类别
