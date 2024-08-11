import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generate moon-shaped data
X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=5)
labels = dbscan.fit_predict(X)

# Extract core samples and their labels
core_samples = X[dbscan.core_sample_indices_]
core_labels = dbscan.labels_[dbscan.core_sample_indices_]

# Fit KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(core_samples, core_labels)

# Predict on the entire dataset
y_pred = knn.predict(X)

# Evaluate and print accuracy
accuracy = accuracy_score(labels, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the DBSCAN results
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')

# Plot the KNN classification results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
plt.title('KNN Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')

plt.savefig("KNC.png")


print("""In short, DBSCAN is a very simple yet powerful algorithm, capable of identifying any
number of clusters, of any shape, it is robust to outliers, and it has just two hyper‐
parameters (eps and min_samples). However, if the density varies significantly across
the clusters, it can be impossible for it to capture all the clusters properly. Moreover,
its computational complexity is roughly O(m log m), making it pretty close to linear
with regards to the number of instances. However, Scikit-Learn’s implementation can
require up to O(m2) memory if eps is large.""")
