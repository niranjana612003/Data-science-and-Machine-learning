import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Original array:\n", arr)


flat_copy = arr.flatten()
print("\nUsing flatten():", flat_copy)


flat_view = arr.ravel()
print("Using ravel():", flat_view)