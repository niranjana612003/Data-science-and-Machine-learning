import numpy as np


matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])


row_vector = np.array([10, 20, 30, 40])


result = matrix + row_vector

print("Original matrix:\n", matrix)
print("\nRow vector to add:", row_vector)
print("\nResult of broadcasting addition:\n", result)

