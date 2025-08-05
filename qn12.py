import numpy as np

arr_1d = np.array([7, 2, 5, 1, 9])
sorted_1d = np.sort(arr_1d)

print("Original 1D array:", arr_1d)
print("Sorted 1D array:", sorted_1d)


arr_2d = np.array([[9, 2, 7],
                   [1, 5, 3],
                   [8, 6, 4]])

sorted_rows = np.sort(arr_2d, axis=1)
print("\nOriginal 2D array:\n", arr_2d)
print("\nRows sorted:\n", sorted_rows)

sorted_columns = np.sort(arr_2d, axis=0)
print("\nColumns sorted:\n", sorted_columns)


arr = np.array([50, 10, 30, 20, 40])
sorted_indices = np.argsort(arr)

print("\nOriginal array:", arr)
print("Indices that would sort the array:", sorted_indices)
print("Sorted array using these indices:", arr[sorted_indices])
