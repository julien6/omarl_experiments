import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
data = np.random.rand(3, 20) * 10  # 3 histories of 10 observation-action couples (as features)

data = np.zeros((3,20))

data[0][0] = 1
data[0][0] = 2

data[1][0] = 1
data[1][0] = 400

# Create a PCA instance with the desired number of components
pca = PCA(n_components=2)

# Fit the PCA model to your data
pca.fit(data)

# Transform the data to the principal components
transformed_data = pca.transform(data)

# Plot the original data
# plt.scatter(data[:, 0], data[:, 1], label='Original Data')

# Plot the transformed data using the principal components
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], label='Transformed Data')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('PCA Example')
plt.show()
