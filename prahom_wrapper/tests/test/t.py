import numpy as np

# Les tableaux concaténés pour chaque type
concat_array_leader = np.array([-2.25000000e-01, -6.90234375e-01,  5.45027087e-01, -2.84635439e-02,
                                -8.03790152e-01, -5.72922606e-01, -1.12007474e+00,  6.88713049e-01,
                                 2.95775863e-01,  2.24732627e-01, -5.40556324e-01, -1.75127408e-01,
                                 3.14609390e-03,  6.43882444e-03, -3.55242467e-01, -1.06707888e-01,
                                -1.24252192e+00, -5.32029884e-01,  4.44293132e-01,  4.26446496e-03,
                                 1.76300219e-02, -2.33744016e-01, -5.15145072e-01,  8.44453437e-02,
                                -6.52313621e-84, -4.05544603e-82, -5.26562500e-01,  1.22644482e-48,
                                -1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00])

concat_array_normal = np.array([ 6.15234375e-01,  3.00000000e-01,  1.89784620e-01, -1.35171432e-01,
                                 -4.48547684e-01, -4.66214717e-01, -7.64832269e-01,  7.95420937e-01,
                                  6.51018330e-01,  3.31440516e-01, -1.85313856e-01, -6.84195202e-02,
                                  3.58388561e-01,  1.13146713e-01,  0.00000000e+00,  0.00000000e+00,
                                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                  0.00000000e+00,  0.00000000e+00, -1.59902604e-01,  1.91153232e-01,
                                  0.00000000e+00,  0.00000000e+00, -5.26562500e-01,  1.22644482e-48,
                                  1.00000000e+00, -1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                                  0.00000000e+00,  0.00000000e+00])

concat_array_good = np.array([-1.68750000e-01,  2.08593938e-49,  8.25382657e-02,  5.59817998e-02,
                              -3.41301330e-01, -6.57367949e-01, -6.57585915e-01,  6.04267706e-01,
                               7.58264684e-01,  1.40287284e-01, -7.80675021e-02, -2.59572752e-01,
                               4.65634916e-01, -7.80065193e-02,  0.00000000e+00,  0.00000000e+00,
                               4.57229168e-02, -2.21153232e-01,  0.00000000e+00,  0.00000000e+00,
                               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                               1.00000000e+00, -1.00000000e+00,  0.00000000e+00,  0.00000000e+00])

normal_leader_adversary_sizes = {
    'self_vel': 2,
    'self_pos': 2,
    'landmark_rel_positions': 10,
    'other_agent_rel_positions': 10,
    'other_agent_velocities': 4,
    'self_in_forest': 2,
    'leader_comm': 4
}

good_sizes = {
    'self_vel': 2,
    'self_pos': 2,
    'landmark_rel_positions': 10,
    'other_agent_rel_positions': 10,
    'other_agent_velocities': 2,
    'self_in_forest': 2
}

# Fonction pour extraire les valeurs
def extract_values(concat_array, sizes):
    extracted_values = {}
    index = 0
    for key, size in sizes.items():
        extracted_values[key] = concat_array[index:index+size]
        index += size
    return extracted_values

# Extraction des valeurs pour chaque type de tailles
extracted_values_leader = extract_values(concat_array_leader, normal_leader_adversary_sizes)
extracted_values_normal = extract_values(concat_array_normal, normal_leader_adversary_sizes)
extracted_values_good = extract_values(concat_array_good, good_sizes)

# Affichage des résultats
print("Extracted values for sizes_leader:")
for key, value in extracted_values_leader.items():
    print(f"{key}: {value}")

print("\nExtracted values for sizes_normal:")
for key, value in extracted_values_normal.items():
    print(f"{key}: {value}")

print("\nExtracted values for sizes_good:")
for key, value in extracted_values_good.items():
    print(f"{key}: {value}")
