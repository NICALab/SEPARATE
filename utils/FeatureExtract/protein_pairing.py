import numpy as np
import networkx as nx

from copy import deepcopy
from itertools import combinations

def calc_distance(latent_features, image_labels, protein_list, pairing_protein_list):
    image_classes = np.max(image_labels)+1
    
    mean_features = dict()
    std_features = dict()
    for image_class in range(image_classes):
        n_data = latent_features[image_labels==image_class].shape[0]
        mean_features[protein_list[image_class]] = np.mean(latent_features[image_labels==image_class].reshape(n_data, -1), 0)
        std_features[protein_list[image_class]] = np.std(latent_features[image_labels==image_class].reshape(n_data, -1), 0)
    
    distance_dictionary = dict()
    distance_matrix = np.zeros((len(pairing_protein_list),len(pairing_protein_list)))
    
    for protein1, protein2 in list(combinations(pairing_protein_list, 2)):
        ##### Project the vector #####
        subtract_vector = mean_features[protein1] - mean_features[protein2]
        subtract_vector_norm = np.linalg.norm(subtract_vector)
        
        ##### feature 1 #####
        n_data1 = latent_features[image_labels==protein_list.index(protein1)].shape[0]
        features1 = latent_features[image_labels==protein_list.index(protein1)].reshape(n_data1, -1)
                
        projected_features1 = list()
        for i in range(n_data1):
            x = np.dot(features1[i] - mean_features[protein2], subtract_vector) / np.sqrt(np.dot(subtract_vector, subtract_vector))
            projected_features1.append(x)
        
        projected_features1 = np.array(projected_features1)
        mean_features1 = np.mean(projected_features1, 0)
        std_features1 = np.std(projected_features1, 0)
        
        ##### feature 2 #####
        n_data2 = latent_features[image_labels==protein_list.index(protein2)].shape[0]
        features2 = latent_features[image_labels==protein_list.index(protein2)].reshape(n_data2, -1)
        
        projected_features2 = list()
        for i in range(n_data2):
            x = np.dot(features2[i] - mean_features[protein2], subtract_vector) / np.sqrt(np.dot(subtract_vector, subtract_vector))
            projected_features2.append(x)
        
        projected_features2 = np.array(projected_features2)
        mean_features2 = np.mean(projected_features2, 0)
        std_features2 = np.std(projected_features2, 0)
        
        ##### feature-based distance using average linkage #####
        average_linkage = list()
        for feature1 in features1:
            for feature2 in features2:
                average_linkage.append(abs(feature1 - feature2))
                
        distance_dictionary[(protein1, protein2)] = np.mean(average_linkage)
        distance_matrix[pairing_protein_list.index(protein1), pairing_protein_list.index(protein2)] = np.mean(average_linkage)
        
    distance_matrix = distance_matrix + distance_matrix.T
    
    return distance_dictionary, distance_matrix

def protein_pairing_helper(G, paired_protein, unpaired_protein):
    G_tmp = deepcopy(G)
    for protein in list(G_tmp.nodes):
        if protein not in unpaired_protein:
            G_tmp.remove_node(protein)

    n_pair = len(unpaired_protein)//2
    tmp_maximum_matching = nx.max_weight_matching(G_tmp, maxcardinality=True)
    
    if len(tmp_maximum_matching) == n_pair:
        return True

def protein_pairing_recursive(scores, pairing_protein_list):
    G = nx.Graph()
    sorted_scores = sorted(scores, key = lambda item: item[1], reverse=True)
    
    pairing_results = list()
    paired_protein = list()
    unpaired_protein = pairing_protein_list
    n_pair = len(unpaired_protein)//2
    
    for score in sorted_scores:
        if set(score[0]).issubset(set(unpaired_protein)):
            G.add_edge(*score[0], weight = score[1])
            
            if len(G.edges) >= n_pair:
                paired_protein_tmp = paired_protein + list(score[0])
                unpaired_protein_tmp = list(set(unpaired_protein) - set(paired_protein_tmp))
                
                if protein_pairing_helper(G, paired_protein_tmp, unpaired_protein_tmp):
                    pairing_results.append(score)
                    
                    pairing_results_tmp, paired_, unpaired_ = protein_pairing_recursive(scores, unpaired_protein_tmp)
                    pairing_results += pairing_results_tmp
                    
                    paired_protein = paired_protein_tmp + paired_
                    unpaired_protein = unpaired_
                    
                    if len(unpaired_) <= 1:
                        break
        else:
            continue
    
    return pairing_results, paired_protein, unpaired_protein

def find_optimal_group_of_pairs(scores, pairing_protein_list):
    print("\n##### Spatial expression pattern guided protein pairing #####")
    
    pairing_results, paired_protein, unpaired_protein = protein_pairing_recursive(scores, pairing_protein_list)
    
    if len(unpaired_protein) != 0:
        print("As the number of proteins are odd, there is \033[93mone unused protein\033[0m: {}".format(unpaired_protein[0]))
    
    print("The minimum feature-based distance is \033[96m{}\033[0m".format(pairing_results[0][1]))
    print("Use the following pairs: ")
    for result in pairing_results:
        print("    ", *result[0], "[\033[96m{}\033[0m]".format(result[1]))
    
    return pairing_results


if __name__=="__main__":
    pass