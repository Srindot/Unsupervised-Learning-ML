import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Correct path construction
image_path = os.path.join("LadyBug.png")

# Load the image
image = imread(image_path)

# Print the shape of the image
print(image.shape)

# Data Preprocessing
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

# Display the segmented image
plt.imshow(segmented_img)
plt.title('Segmented Image')
plt.axis('off')
plt.savefig("segmentedimage.png")


