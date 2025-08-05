import numpy as np

arr = np.array([[ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16]])

print("Original 4x4 array:\n", arr)

center_submatrix = arr[1:3, 1:3]
print("\n2x2 center submatrix:\n", center_submatrix)