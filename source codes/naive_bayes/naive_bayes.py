import math


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train，求出训练集X_train各个特征的均值和标准差
    def summarize(self, X_train):
        summaries = [(self.mean(data), self.stdev(data)) for data in zip(*X_train)]
        return summaries

    # 分类别 求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for x, label in zip(X, y):
            data[label].append(x)  # 将相同标签的数据写到相同的字典项中（这里的字典项为列表）
        self.model = {label: self.summarize(value) for label, value in data.items()}

    def calculate_probabilities(self, input_data):  # 计算概率
        probabilities = {}  # 空字典
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):  # 对样本的各个特征分量计算其概率密度，然后再连乘在一起，得到最终的联合概率（密度）
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    # 类别预测 # 先根据测试样本计算出其属于各个类别的概率，再排序，并取出最后一个（概率值最大），最后再获得其标签
    def predict(self, X_test):
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):  # 准确率 accuracy
        right = 0
        for X, y in zip(X_test, y_test):  # 统计测试数据预测值与真实标签相同的个数，获得准确率
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))
