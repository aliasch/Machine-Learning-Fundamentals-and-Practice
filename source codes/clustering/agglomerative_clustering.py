import heapq
import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=1):
        self.k = n_clusters

    def fit_transform(self, X):
        m, n = X.shape
        # C 是一个字典，记录当前聚成的所有类，初始时为m个类cluster。
        # C的键id为类的编号，对应的值是该类中的全体样本的编号，为一个列表。C={1:[1,2,3],2:[5,7,10]}
        # centers字典，键为类cluster编号，值为属于该类的中心坐标，centers={1:coordinate,...}
        C, centers = {}, {}
        assignments = np.zeros(m)  # 用于记录每个样本所属类的编号，初始化为每一个样本编号i，0<=i<m，属于类i下面for循环对各个变量做初始化。
        for id in range(m):
            C[id] = [id]
            centers[id] = X[id]
            assignments[id] = id  # 下标为id的元素，其初始类别为id
        H = []  # 用作优先队列的数组H，其元素为一个元组(d,[i,j])，d表示样本编号i和j的距离。
        for i in range(m):  # 计算任意两个样本之间的距离，并将其加入到优先队列中
            for j in range(i + 1, m):
                d = np.linalg.norm(X[i] - X[j], 2)
                heapq.heappush(H, (d, [i, j]))
        new_id = m
        while len(C) > self.k:
            distance, [id1, id2] = heapq.heappop(H)
            if id1 not in C or id2 not in C:  # 说明某个类别id1或id2已经被合并进其他类别中了，其编号就移除了
                continue
            # C是一个字典，C[id1]是属于id1类的样本集编号,将这些编号拼接起来，
            # 即C[id1]是样本X[i]的编号的集合
            C[new_id] = C[id1] + C[id2]
            # 将归并为同一个类的样本的下标i都设置为他们属于新的类new_id，即样本i的类cluster为new_id
            for i in C[new_id]:
                assignments[i] = new_id
            del C[id1], C[id2], centers[id1], centers[id2]  # 删除已经被合并的类及其中心
            new_center = sum(X[C[new_id]]) / len(C[new_id])  # 计算得出新类cluster的中心坐标
            for id in centers:
                center = centers[id]  # 类cluster为id的中心坐标为center(同一类的多个样本的中心坐标）
                d = np.linalg.norm(new_center - center, 2)  # 计算(调整)新加入的中心坐标跟原来的那些中心的距离
                heapq.heappush(H, (d, [id, new_id]))
            centers[new_id] = new_center
            new_id += 1
        return np.array(list(centers.values())), assignments  # 返回各类(cluster)中心坐标和各元素所属的类别
