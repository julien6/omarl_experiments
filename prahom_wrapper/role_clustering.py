import copy
from typing import Any, Dict, List, Union
import numpy
import json
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from prahom_wrapper.history_model import action, observation


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
                lcs = longest_common_subsequence(s1, s2)
                if (len(lcs) > len(max_lcs)):
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


def compute_role_clustering(joint_history: Dict[str, List[Union[observation, action]]], generate_figure: bool = False):
    """Operates a hierachical sequence clustering based on the common longest sequence.
    A dendrogram may be generated as a complementary help.
    """
    # Matrice de similarité

    sequences = list(joint_history.values())

    # sequences = [s1, s2, s3, s4, s5, s6, s6]
    # result = longest_common_subsequence_multiple(sequences)
    # result, s1, s2 = find_two_subsequences_with_best_lcs(sequences)
    # print("La sous-séquence commune la plus grande est :", result, " pour les sequences: ", s1, " et ", s2)

    distance_matrix = [[0] * len(sequences) for _ in sequences]
    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences):
            distance_matrix[i][j] = sequence_dist(seq1, seq2)

    distance_matrix = np.array(distance_matrix)

    # print(distance_matrix)

    # Calcul du clustering hiérarchique avec la méthode de liaison complète
    Z = linkage(distance_matrix, method='complete')

    # Z = np.array([[2,  7,  0,  2],
    #               [0,  9,  0,  2],
    #               [1,  6,  0,  2],
    #               [5, 10,  0,  3],
    #               [11, 12,  0,  4],
    #               [4,  8,  0,  2],
    #               [14, 15,  0,  6],
    #               [13, 16,  0,  9],
    #               [3, 17,  1, 10]], dtype=float)

    Z[:, 2] = np.arange(1., len(Z)+1)

    # labels = [str(len(Z)+1+ind)+'='+str(Z[ind, 0].astype(int))+'+' +
    #           str(Z[ind, 1].astype(int)) for ind in range(len(Z))]

    labels = []
    leaves_seqs = [str(x) for x in joint_history.keys()]
    for ind in range(len(Z)):
        seq1 = sequences[Z[ind, 0].astype(int)]
        seq2 = sequences[Z[ind, 1].astype(int)]
        lcs = longest_common_subsequence(seq1, seq2)
        labels.append(str(len(Z)+1+ind)+'=lcs('+str(Z[ind, 0].astype(int))+',' +
                      str(Z[ind, 1].astype(int))+')=' + str(lcs) + '\ndist=' + str(sequence_dist(seq1, seq2)))
        sequences.append(lcs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    dn = dendrogram(Z, ax=ax, labels=leaves_seqs)
    ii = np.argsort(np.array(dn['dcoord'])[:, 1])
    for j, (icoord, dcoord) in enumerate(zip(dn['icoord'], dn['dcoord'])):
        x = 0.5 * sum(icoord[1:3])
        y = dcoord[1]
        ind = np.nonzero(ii == j)[0][0]
        ax.annotate(labels[ind], (x, y), va='top', ha='center')

    SMALL_SIZE = 8
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)
    matplotlib.rcParams.update({'font.size': SMALL_SIZE})

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./role_clustering.png')


# s1 = ["o0", "a0", "o1", "a0", "o1", "a0", "o1"]
# s2 = ["o0", "a1", "o1", "a1", "o1", "a0", "o1"]
# s3 = ["o0", "a1", "o1", "a1", "o1", "a1", "o1"]
# perfect_policy = [5, 0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 6, 0, 0, 0, 5, 0, 0, 4, 0, 0, 4,
#                     0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 6, 0, 0, 0, 5, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 6]
# joint_history = {
#     "role_0": s1,
#     "role_1": s2,
#     "role_2": s3
# }

joint_history = {'agent_0': [5, 2, 2, 2, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'agent_1': [0, 0, 0, 0, 0, 0, 0, 5, 4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0], 'agent_2': [0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 1, 1, 1, 6]}

def generate_r_clustering():

    joint_history = {'agent_0': [5, 2, 2, 2, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'agent_1': [0, 0, 0, 0, 0, 0, 0, 5, 4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0], 'agent_2': [0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 1, 1, 1, 6]}

    compute_role_clustering(joint_history=joint_history)
