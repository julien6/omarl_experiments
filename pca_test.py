import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# # Generate some example data
# np.random.seed(42)
# # 3 histories of 10 observation-action couples (as features)
# data = np.random.rand(3, 20) * 10

# generated data
actions = {'piston_0': [1, 2, 1, 1, 0, 0, 1], 'piston_1': [1, 1, 1, 1, 1, 1, 1], 'piston_2': [
    2, 1, 0, 2, 1, 0, 1], 'piston_3': [1, 1, 1, 1, 2, 0, 1], 'piston_4': [1, 2, 1, 1, 1, 2, 1]}

agents = list(actions.keys())

data = [actions for _, actions in actions.items()]

data = np.array(data)

# Create a PCA instance with the desired number of components
pca = PCA(n_components=2)

# Fit the PCA model to your data
pca.fit(data)

# Transform the data to the principal components
transformed_data = pca.transform(data)

# Plot the transformed data using the principal components
for index, agent in enumerate(agents):
    plt.scatter(transformed_data[index][0],
                transformed_data[index][1], label=agent)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('PCA Example')
plt.show()
