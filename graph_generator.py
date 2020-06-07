import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def generate_random_tree(number_of_nodes: int) -> nx.Graph :
    return nx.random_tree(n=number_of_nodes)


def create_node_distance_matrix(tree: nx.Graph) -> np.array:
    is_leaf = [False] * len(tree.nodes)
    for node in tree.nodes:
        if tree.degree[node] == 1:
            is_leaf[node] = True
    leaf_indices = np.where(is_leaf)[0]
    leaf_distance_matrix = np.ndarray(shape=(len(leaf_indices), len(leaf_indices)))
    for i in range(0, len(leaf_indices)):
        for j in range(0, len(leaf_indices)):
            distance = nx.shortest_path_length(tree,leaf_indices[i], leaf_indices[j])
            leaf_distance_matrix[i, j] = distance
    return leaf_distance_matrix


def generate_example(number_of_nodes: int) -> np.array:
    tree = generate_random_tree(number_of_nodes)
    return create_node_distance_matrix(tree)
