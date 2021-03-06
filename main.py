import numpy as np
from glob import glob
import pkg_resources.py2_warn
# import matplotlib.pyplot as plt
# import networkx as nx
# import pandas as pd

matrix_1 = np.array([
    [0, 3, 2],
    [4, 0, 2],
    [2, 2, 0]
])

def create_upper_triangular_matrix(matrix: np.array) -> np.array:
    # Tworzymy macierz górnotrójkątną żeby uniknąć niejednoznaczności
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j <= i:
                matrix[i][j] = 0
    return matrix


def find_smallest_distance_coordinates_in_matrix(matrix: np) -> (int, int):
    # Maskujemy w macierzy wartości, dzięki czemu przy wielu najmniejszych wartościach łatwo możemy po nich iterować
    tmp = np.ma.MaskedArray(matrix, matrix <= 0)
    return np.unravel_index(tmp.argmin(), tmp.shape)


def create_distances_matrix_to_merge_vertex(matrix: np, a, b) -> np.array:
    distances_to_merge_vertex = np.zeros(len(matrix))
    for i in range(len(matrix)):
        if not i == a and not i == b:
            if a > i and b > i:
                distances_to_merge_vertex[i] = (matrix[i][a] + matrix[i][b] - matrix[a][b]) / 2
                distances_to_merge_vertex[a] = matrix[i][a] - distances_to_merge_vertex[i]
                distances_to_merge_vertex[b] = matrix[i][b] - distances_to_merge_vertex[i]
            if a < i < b:
                distances_to_merge_vertex[i] = (matrix[a][i] + matrix[i][b] - matrix[a][b]) / 2
                distances_to_merge_vertex[a] = matrix[a][i] - distances_to_merge_vertex[i]
                distances_to_merge_vertex[b] = matrix[i][b] - distances_to_merge_vertex[i]
            if a < i and b < i:
                distances_to_merge_vertex[i] = (matrix[a][i] + matrix[b][i] - matrix[a][b]) / 2
                distances_to_merge_vertex[a] = matrix[a][i] - distances_to_merge_vertex[i]
                distances_to_merge_vertex[b] = matrix[b][i] - distances_to_merge_vertex[i]
    return distances_to_merge_vertex


def check_if_distances_are_correct(matrix: np, a, b, distances_to_merge_vertex) -> [True, False]:
    for i in range(len(matrix)):
        if not i == a and not i == b:
            if a > i and b > i:
                if not matrix[i][a] == distances_to_merge_vertex[a] + distances_to_merge_vertex[i] \
                        or not matrix[i][b] == distances_to_merge_vertex[b] + distances_to_merge_vertex[i]:
                    return False
            if a < i < b:
                if (not matrix[a][i] == distances_to_merge_vertex[a] + distances_to_merge_vertex[i]) \
                        or (not matrix[i][b] == distances_to_merge_vertex[b] + distances_to_merge_vertex[i]):
                    return False
            if a < i and b < i:
                if not matrix[a][i] == distances_to_merge_vertex[a] + distances_to_merge_vertex[i] \
                        or not matrix[b][i] == distances_to_merge_vertex[b] + distances_to_merge_vertex[i]:
                    return False
    return True


def merge_subgraphs(new_subgraph, first_subgraph, first: bool):
    merged_subgraph = new_subgraph
    add_edge = False
    if not type(first_subgraph) == type(0):
        new_subgraph_merge_idx = np.unravel_index(new_subgraph.argmax(), new_subgraph.shape)[0]
        merge_idx = np.unravel_index(first_subgraph.argmax(), first_subgraph.shape)
        if first:
            if new_subgraph_merge_idx == 0:
                add_edge = True
            deleted_column = new_subgraph[0, :]
            deleted_column = np.delete(deleted_column, 0, 0)
            new_subgraph = np.delete(new_subgraph, 0, 0)
            new_subgraph = np.delete(new_subgraph, 0, 1)
        else:
            if new_subgraph_merge_idx == len(new_subgraph[0])-1:
                add_edge = True
            deleted_column = new_subgraph[:, len(new_subgraph[0]) - 1]
            new_subgraph = np.delete(new_subgraph, len(new_subgraph[0]) - 1, 0)
            new_subgraph = np.delete(new_subgraph, len(new_subgraph[0]) - 1, 1)
        first_subgraph[merge_idx] = 0
        tmp = np.zeros((len(new_subgraph[0]), len(first_subgraph[0])))
        tmp = np.concatenate((first_subgraph, tmp))
        tmp2 = np.zeros((len(first_subgraph[0]), len(new_subgraph[0])))
        tmp2 = np.concatenate((tmp2, new_subgraph))
        tmp = np.concatenate((tmp, tmp2), axis=1)
        for i in range(len(deleted_column)):
            if (deleted_column[i] == 1):
                tmp[int(merge_idx[0])][len(first_subgraph)+i] = 1
        if add_edge:
            tmp[merge_idx[0]][merge_idx[0]] = 2
        merged_subgraph = tmp
    return merged_subgraph


def count_number_of_leaves(matrix):
    number_of_leaves = 0
    for i in range(len(matrix)):
        column = matrix[:, i]
        row = matrix[i, :]
        if np.sum(column) + np.sum(row) == 1:
            number_of_leaves += 1
    return number_of_leaves


def make_graph_from_leaves(matrix):
    number_of_leaves = len(matrix[0])
    subgraphs_matrix = [0] * len(matrix[0])
    b = matrix
    bad_pair = False

    while b.shape[0] > 1:

        if not bad_pair:
            upper_triangular_matrix = create_upper_triangular_matrix(b)
            new_masked_matrix = np.ma.MaskedArray(upper_triangular_matrix, upper_triangular_matrix <= 0)

        smallest_distance_idx = find_smallest_distance_coordinates_in_matrix(new_masked_matrix)

        distances_to_merge_vertex = create_distances_matrix_to_merge_vertex(upper_triangular_matrix, smallest_distance_idx[0],
                                                                            smallest_distance_idx[1])

        distances_to_merge_vertex = distances_to_merge_vertex.astype(int)
        if (check_if_distances_are_correct(upper_triangular_matrix, smallest_distance_idx[0], smallest_distance_idx[1],
                                           distances_to_merge_vertex)):
            smallest_distance = int(new_masked_matrix[smallest_distance_idx[0]][smallest_distance_idx[1]])
            distances_to_merge_vertex_difference = (distances_to_merge_vertex[smallest_distance_idx[0]] - distances_to_merge_vertex[smallest_distance_idx[1]] + smallest_distance)/2

            new_subgraph = np.zeros((smallest_distance+1, smallest_distance+1))
            first_subgraph = subgraphs_matrix[smallest_distance_idx[0]]
            second_subgraph = subgraphs_matrix[smallest_distance_idx[1]]

            for i in range(smallest_distance):
                new_subgraph[i][i+1] = 1
            new_subgraph[int(distances_to_merge_vertex_difference)][int(distances_to_merge_vertex_difference)] = 2
            merged_subgraph = merge_subgraphs(new_subgraph, first_subgraph, True)
            merged_subgraph = merge_subgraphs(merged_subgraph, second_subgraph, False)
            # Stworzenie macierzy B' w której X i Y sa usunenie a Z jest dodane
            new_masked_matrix.mask = np.ma.nomask
            b = np.hstack((new_masked_matrix, np.atleast_2d(distances_to_merge_vertex).T))
            b = np.vstack((b, [0] * b.shape[1]))
            b = np.delete(b, [smallest_distance_idx[0], smallest_distance_idx[1]], 0)
            b = np.delete(b, [smallest_distance_idx[0], smallest_distance_idx[1]], 1)
            b.mask = np.ma.nomask
            subgraphs_matrix.pop(max(smallest_distance_idx[0], smallest_distance_idx[1]))
            subgraphs_matrix.pop(min(smallest_distance_idx[0], smallest_distance_idx[1]))
            subgraphs_matrix.append(merged_subgraph)

            bad_pair = False

        else:
            new_masked_matrix[smallest_distance_idx[0], smallest_distance_idx[1]] = np.ma.masked
            b = new_masked_matrix
            bad_pair = True
            if False not in new_masked_matrix.mask:
                # in this step we can return -1, non any pair from input create valid pair
                return -1

    tmp_idx = np.unravel_index(merged_subgraph.argmax(), merged_subgraph.shape)
    merged_subgraph[tmp_idx[0]][tmp_idx[1]] = 0
    if not count_number_of_leaves(merged_subgraph) == number_of_leaves:
        return -1
    return merged_subgraph


def run():

    # look for input.csv
    input_files = glob('*.txt') + glob('*.csv')
    output_file = open('result.txt', 'w')
    # res = make_graph_from_leaves(matrix_1)
    # print(res)
    try:
        for file in input_files:
            if file != 'requirements.txt' and file != "result.txt":
                rows_to_skip = 0
                if 'csv' in file:
                    rows_to_skip = 1
                input_matrix = np.loadtxt(file, delimiter=",", skiprows=rows_to_skip)
                res = make_graph_from_leaves(input_matrix)
                output_file.write(f"\n*******************************   OUTPUT DLA PLIKU {file}    *******************************\n")
                neighbour_list = []
                for i in range(len(res[0])):
                    for j in range(i, len(res[0])):
                        res[j][i] = res[i][j]
                output_file.write("Macierz sąsiedztwa:\n")
                output_file.write(str(res))
                output_file.write("\nLista sąsiedztwa: \n")
                for i in range(len(res[0])):
                    list = []
                    for j in range(len(res[0])):
                        if res[i][j] == 1:
                            list.append(j)
                    neighbour_list.append(list)

                    output_file.write(str(i) + ": " + str(list) + "\n")



                # G = nx.from_numpy_array(res)
                # nx.draw(G)
                # plt.show()
        output_file.close()
    except OSError:
        print("Plik wejsciowy nie znaleziony. Prosze stworzyc plik input.csv w tym katalogu")


if __name__ == '__main__':
    run()
