import copy
from typing import List
import numpy
import json
import matplotlib.pyplot as plt
import csv
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# s1 = [1,2,0,1,2,4,1,3,4,3,2,1,5,4]
# s2 = [0,0,1,7,2,3,4,9,5,3,9,7]
# s3 = [1,8,2,3,6,4,11,5,0]
# s4 = [0,1,41,2,3,6,4,5,14]

# sequences = [s1, s2, s3, s4]

# s = [1,2,3,4,5]

def longest_common_subsequence(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruction de la sous-séquence commune
    i, j = m, n
    common_sequence = []
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            common_sequence.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return common_sequence[::-1]

def longest_common_subsequence_multiple(sequences):
    if not sequences:
        return []

    common_sequence = sequences[0]
    for seq in sequences[1:]:
        common_sequence = longest_common_subsequence(common_sequence, seq)

    return common_sequence

def find_two_subsequences_with_best_lcs(sequences):

    max_lcs = []
    seq1 = []
    seq2 = []
    for i, s1 in enumerate(sequences):
        for j, s2 in enumerate(sequences):
            if i != j:
                lcs = longest_common_subsequence(s1,s2)
                if(len(lcs) > len(max_lcs)):
                    max_lcs = lcs
                    seq1 = s1
                    seq2 = s2

    return max_lcs, seq1, seq2

def find_subsequences_with_best_lcs(sequences: List[List[int]]):
    seqs = copy.copy(sequences)
    lcs = []
    seq1 = None
    seq2 = None
    for _ in range(len(seqs) - 1):
        lcs, seq1, seq2 = find_two_subsequences_with_best_lcs(seqs)
        seqs.remove(seq1)
        seqs.remove(seq2)
        seqs.append(lcs)
    return lcs, seq1, seq2

def sequence_dist(seq1, seq2):
    lcs = longest_common_subsequence(seq1, seq2)
    return 1 - (2 * len(lcs) / (len(seq1)+len(seq2)))

actions = {'piston_0': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_1': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_2': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0], 'piston_3': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 0], 'piston_4': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 0], 'piston_5': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 0], 'piston_6': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 0], 'piston_7': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_8': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_9': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_10': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_11': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_12': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_13': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_14': [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_15': [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_16': [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_17': [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_18': [1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], 'piston_19': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}

acts = [a for p_name, a in actions.items()]

s1 = [1,2,4,5,6,0,4,0,1,0,0]
s2 = [1,2,3,4,5,6,0,0,0,0,0]
s3 = [1,8,2,3,6,4,1,5,0,0,0]
s4 = [0,1,4,1,2,3,6,4,5,1,9]
s5 = [5,4,3,2,1,0,1,2,3,4,5]
s6 = [5,0,0,4,2,2,1,3,7,4,2]

# Matrice de similarité

# sequences = acts
sequences = [s1, s2, s3, s4, s5, s6, s6]
# result = longest_common_subsequence_multiple(sequences)
# result, s1, s2 = find_two_subsequences_with_best_lcs(sequences)
# print("La sous-séquence commune la plus grande est :", result, " pour les sequences: ", s1, " et ", s2)

distance_matrix = [[0] * len(sequences) for _ in sequences]
for i, seq1 in enumerate(sequences):
    for j, seq2 in enumerate(sequences):
        distance_matrix[i][j] = sequence_dist(seq1, seq2)

distance_matrix = np.array(distance_matrix)

print(distance_matrix)

# Calcul du clustering hiérarchique avec la méthode de liaison complète
Z = linkage(distance_matrix, method='complete')

print(Z)

# Affichage du dendrogramme
plt.figure(figsize=(10, 5))
dendrogram(Z)#, labels=["s1", "s2", "s3", "s4", "s5", "s6"])
plt.title('Dendrogramme')
plt.xlabel('Index des échantillons')
plt.ylabel('Distance')
plt.show()
