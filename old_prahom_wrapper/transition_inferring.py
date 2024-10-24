import itertools
import random
from typing import Any
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx


def k_mean(data: np.ndarray):

    # Reshape the data into a 2D array
    # reshaped_data = data.reshape(data.shape[0], -1)  # Reshape to (21, 100)

    reshaped_data = data

    # Initialize and fit the K-means model
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(reshaped_data)

    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    print("Cluster centers:", cluster_centers)
    print("Labels:", labels)

    # Visualize the data and the clusters
    plt.scatter(reshaped_data[:, 0],
                reshaped_data[:, 1], c=labels, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker='x', color='red', s=100, label='Cluster centers')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def pca(data: np.ndarray):

    # Create a PCA instance with the desired number of components
    pca = PCA(n_components=2)

    # Fit the PCA model to your data
    pca.fit(data)

    # Transform the data to the principal components
    transformed_data = pca.transform(data)

    # Plot the transformed data using the principal components
    for index in range(data.shape[0]):
        plt.scatter(transformed_data[index][0],
                    transformed_data[index][1], label="state_" + str(index))

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('States PCA')
    plt.show()


def graph():

    # Créer un graphe dirigé
    G = nx.DiGraph()
    llab = {}

    # Ajouter les nœuds à la chaîne
    num_nodes = 7
    for i in range(num_nodes):
        G.add_node(i)

    t_t = []
    # Ajouter les arêtes entre les nœuds pour former la chaîne
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
        t_t += [(i, i+1)]
        llab[(i, i+1)] = " "

    l = []
    for i in range(num_nodes):
        for j in range(3):
            G.add_node(str(i)+str(j))
            G.add_edge(str(i)+str(j), i)
            l += [(str(i)+str(j), i)]
            llab[(str(i)+str(j), i)] = "dist=" + str(round(random.random(),3))

     # Draw the graph
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, ax=ax,nodelist=range(7))
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=t_t, edge_color="blue")

    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=l, ax=ax, arrows=False)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=llab)

    # plt.show()
    plt.savefig("./transition_goals.png")


states1 = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0, 0, 4, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 5, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 3, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 3, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [
    0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 3, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 3, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 1, 3, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 1, 1, 3, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 1, 1, 2, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 3, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

states2 = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0, 0, 4, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 5, 2, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 3, 1, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 3, 1, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [
    0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 3, 1, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 3, 1, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 1, 3, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 1, 1, 3, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 2, 0], [0, 4, 1, 1, 1, 1, 1, 2, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 3, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 2, 1, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 4, 0], [0, 1, 0, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 2, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 1, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]


states1 = [list(itertools.chain.from_iterable(state)) for state in states1]

states2 = [list(itertools.chain.from_iterable(state)) for state in states2]

states = states1 + states2

states = np.asarray(states)


# k_mean(states)

# pca(states)

graph()
