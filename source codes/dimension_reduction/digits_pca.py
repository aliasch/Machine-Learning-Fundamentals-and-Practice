import matplotlib.pyplot as plt
from pca import PCA
from sklearn.datasets import load_digits
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

digits = load_digits()
X = digits.data
y = digits.target

model = PCA(n_components = 10)
Z = model.fit_transform(X)
X_recovered = model.inverse_transform(Z).astype(int)

plt.figure(0)
plt.subplot(1,2,1)
plt.imshow(X[100].reshape(8,8))
plt.title('原始图像')
plt.subplot(1,2,2)
plt.title('降维后重构的图像')
plt.imshow(X_recovered[100].reshape(8,8))
plt.figure(1)
plt.scatter(Z[:,0], Z[:,1], c = y)
plt.show()