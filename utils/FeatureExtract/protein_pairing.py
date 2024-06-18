import numpy as np
import networkx as nx

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

def protein_pairing(G, n_pair, results=list(), used_protein=set()):
    tmp_maximum_matching = nx.max_weight_matching(G, maxcardinality=True)
    
    if len(tmp_maximum_matching) == n_pair:
        results.append(min([(edge, G.edges[list(edge)]["weight"]) for edge in tmp_maximum_matching], key = lambda x : x[1]))
        used_protein.update(set(results[-1][0]))
        
        G.remove_node(results[-1][0][0])
        G.remove_node(results[-1][0][1])
        
        if n_pair > 1:
            results, used_protein = protein_pairing(G, n_pair-1, results, used_protein)
                    
    return results, used_protein

def find_optimal_group_of_pairs(scores, pairing_protein_list):
    print("\n##### finding optimal group of pairs #####")
    G = nx.Graph()
    sorted_scores = sorted(scores, key = lambda item: item[1], reverse=True)
    n_pair = len(pairing_protein_list)//2
    for score in sorted_scores[:n_pair]:
        G.add_edge(*score[0], weight = score[1])
    # breakpoint()
    for i in range(n_pair+1, len(sorted_scores)+1):
        pair_results, used_protein = protein_pairing(G, n_pair)
        if len(pair_results) == 0:
            G.add_edge(*sorted_scores[i][0], weight = sorted_scores[i][1])
        else:
            break
    # print the result
    if len(pairing_protein_list) != len(used_protein):
        unused_protein = list(set(pairing_protein_list) - set(used_protein))[0]
        print("As the number of antibodies are odd, there is \033[93mone unused antibody\033[0m: {}".format(unused_protein))
    print("The minimum score is \033[96m{}\033[0m".format(pair_results[0][1]))
    print("Use the following pairs: ")
    for pair_result in pair_results:
        print("    ",*pair_result[0])
    return None


if __name__=="__main__":
    pass