import numpy as np
from svm_smo import SVM


class KernelSVM(SVM):

    def __init__(self, kernel=None): # 若kernel没有传入参数，就是不使用核函数，否则就是使用传入的核函数。
        super().__init__()
        self.kernel = kernel

    def get_K(self, X1, X2):
        if self.kernel is None:
            return X1.dot(X2.T)
        m1, m2 = len(X1), len(X2)
        K = np.zeros((m1, m2))   # 初始化矩阵K
        for i in range(m1):
            for j in range(m2):
                K[i][j] = self.kernel(X1[i], X2[j])
        return K  # 返回样本的内积构成的矩阵或核函数构成的矩阵

    def fit(self, X, y, N=10):
        K = self.get_K(X, X)  # 获得K矩阵
        self.smo(X, y, K, N)  # 调用smo算法，传入样本X,标签y,样本核函数的矩阵K,坐标下降算法搜索的轮次N。
        self.X_train = X  # 预测样本分类时需要用到训练数据，所以将训练数据X存到X_train起来
        self.y_train = y

    def predict(self, X):
        K = self.get_K(X, self.X_train)  # 此处用到了上面保存的X_train
        return np.sign(K.dot(self.Lambda * self.y_train) + self.b)  # 分类判断
