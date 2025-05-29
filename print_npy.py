import numpy as np
import matplotlib.pyplot as plt

data = np.load("clustered_dataset_l1.npy")
labels = np.load("clustered_memberships_l1.npy")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("l1-KMeans Clustering Result (first 2 features)")
plt.colorbar(scatter, label="Cluster Label")
plt.savefig("kmeans_result_L1Dist.png")
print("图片已保存为 kmeans_result.png")

data = np.load("clustered_dataset_l2.npy")
labels = np.load("clustered_memberships_l2.npy")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("l2-KMeans Clustering Result (first 2 features)")
plt.colorbar(scatter, label="Cluster Label")
plt.savefig("kmeans_result_L2Dist.png")
print("图片已保存为 kmeans_result.png")

data = np.load("clustered_dataset_linf.npy")
labels = np.load("clustered_memberships_linf.npy")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("linf-KMeans Clustering Result (first 2 features)")
plt.colorbar(scatter, label="Cluster Label")
plt.savefig("kmeans_result_LInfDist.png")
print("图片已保存为 kmeans_result.png")