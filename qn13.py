import numpy as np

arr = np.array([5, 12, 7, 15, 3, 20])

result = np.where(arr > 10, 1, 0)

print("Original array:", arr)
print("After np.where():", result)


