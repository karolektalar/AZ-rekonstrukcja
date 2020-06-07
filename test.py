import numpy as np
from glob import glob

input_files = glob('result.csv')
for file in input_files:
    if file != 'requirements.txt' and file != "result.txt":
        rows_to_skip = 0
        if 'csv' in file:
            rows_to_skip = 1
        input_matrix = np.loadtxt(file, delimiter=",", skiprows=rows_to_skip)
        result_matrix = np.zeros((40, 3))
        j = 0
        for i in range(len(input_matrix)):
            if i % 10 == 0:
                result_matrix[j][0] = input_matrix[i][0]
                tmp_matrix = input_matrix[i:i+9]
                var = np.var(tmp_matrix, axis=0)
                avg = np.average(tmp_matrix, axis=0)
                result_matrix[j][1] = var[1]
                result_matrix[j][2] = avg[1]
                j += 1
        print(result_matrix)


