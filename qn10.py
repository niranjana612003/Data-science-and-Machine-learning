import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print("Original array:", arr)


last_element = arr[-1]
print("Last element (using negative indexing):", last_element)


filtered = arr[arr > 25]
print("Elements greater than 25 (boolean indexing):", filtered)

