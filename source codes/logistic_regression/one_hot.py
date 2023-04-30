import numpy as np


def one_hot_encoding(y):
    m = len(y)
    k = np.max(y) + 1
    result = np.zeros([m, k])
    for i in range(m):
        result[i, y[i]] = 1
    return result
# 10分类测试
y = [4, 2, 3, 6, 5]
result = one_hot_encoding(y)
print(result)
