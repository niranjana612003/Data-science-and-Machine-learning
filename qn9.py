import numpy as np

arr = np.arange(12)
print("Original 1D array:", arr)


reshaped_3x4 = arr.reshape(3, 4)
print("\nReshaped to 3x4:\n", reshaped_3x4)

reshaped_2x6 = arr.reshape(2, 6)
print("\nReshaped to 2x6:\n", reshaped_2x6)

try:
    arr.reshape(5, 3)
except ValueError as e:
    print("\nError when reshaping to 5x3:", e)
