import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使用USPS的数字数据集，并将这些数据集进行归一化，使得其所有像素值都在(0,1)之间
X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X = MinMaxScaler().fit_transform(X)
# 基本想法是，从带噪图像中学习一个PCA模型（带和不带核），然后使用这些模型重建图片及对图片去噪
# 将数据集分成训练集1000个样本，测试集100个样本。这些样本是无噪声的，以它们为基础，评估去噪方法的效率。
# 在无噪声图片的基础上，认为添加高斯噪声。这个程序的想法就是，展示可以通过从无损坏的图像中学到一个PCA模型，
# 然后利用这个模型，对损坏的图像进行去噪。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1_000, test_size=100)
# 生成带噪声的训练和测试样本
rng = np.random.RandomState(0)
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise
noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise
# 绘制图像的辅助函数定性评估图像重建效果
def plot_digits(X, title):
    """Small helper function to plot 100 digits."""
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)
# 使用MSE定量评估图像重建效果
plot_digits(X_test, "未损坏测试图像")
plot_digits(X_test_noisy, f"有噪声测试图像\n MSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}")
# 学习PCA基，包括线性PCA和高斯核PCA
from sklearn.decomposition import PCA, KernelPCA
pca = PCA(n_components=32)
kernel_pca = KernelPCA(n_components=400, kernel="rbf", gamma=1e-3,
                       fit_inverse_transform=True, alpha=5e-3)
pca.fit(X_train_noisy)
_ = kernel_pca.fit(X_train_noisy)
# 转换和重建测试有噪声的图像。因为PCA设置将特征降维了，因此，将得到原始数据集的一个近似值。
# 通过最少地删除解释PCA中方差的分量，希望能消除噪声。并且对比线性PCA和核PCA，
# 并期待高斯核有更好的结果，因为使用了非线性内核来学习PCA
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test_noisy))
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))
plot_digits(X_reconstructed_pca,
            f"PCA 重建\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}")
plot_digits(X_reconstructed_kernel_pca,
            f"Kernel PCA 重建\nMSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}")
plt.show()