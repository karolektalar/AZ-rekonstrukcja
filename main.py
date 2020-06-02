import numpy as np

matrix_1 = np.array([
    [0, 2, 2, 2],
    [2, 0, 2, 2],
    [2, 2, 0, 2],
    [2, 2, 2, 0],
])


def create_upper_triangular_matrix(matrix: np.array) -> np.array:
    # Tworzymy macierz górnotrójkątną żeby uniknąć niejednoznaczności
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j <= i:
                matrix[i][j] = 0
    return matrix


def find_smallest_distance_coordinates_in_matrix(matrix: np) -> int:
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
                if not matrix[a][i] == distances_to_merge_vertex[a] + distances_to_merge_vertex[i] \
                        or not matrix[i][b] == distances_to_merge_vertex[b] + distances_to_merge_vertex[i]:
                    return False
            if a < i and b < i:
                if not matrix[a][i] == distances_to_merge_vertex[a] + distances_to_merge_vertex[i] \
                        or not matrix[b][i] == distances_to_merge_vertex[b] + distances_to_merge_vertex[i]:
                    print(distances_to_merge_vertex[a] + distances_to_merge_vertex[i])
                    print(distances_to_merge_vertex[b] + distances_to_merge_vertex[i])
                    return False
    return True


def make_graph_from_leaves(matrix):
    new_masked_matrix = create_upper_triangular_matrix(matrix)
    new_masked_matrix = np.ma.MaskedArray(new_masked_matrix, new_masked_matrix <= 0)
    smallest_distance_idx = find_smallest_distance_coordinates_in_matrix(new_masked_matrix)
    distances_to_merge_vertex = create_distances_matrix_to_merge_vertex(new_masked_matrix, smallest_distance_idx[0],
                                                                        smallest_distance_idx[1])
    if (check_if_distances_are_correct(new_masked_matrix, smallest_distance_idx[0], smallest_distance_idx[1],
                                       distances_to_merge_vertex)):
        print("true")
    else:
        print("false")


if __name__ == '__main__':
    make_graph_from_leaves(matrix_1)
