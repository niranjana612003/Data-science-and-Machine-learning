import sys
import time
import numpy as np

python_list = [i for i in range(1000000)]


list_size = sys.getsizeof(python_list)
list_element_size = sys.getsizeof(python_list[0])

print(f"Memory size of Python list (with overhead): {list_size} bytes")
print(f"Memory size of one element in Python list: {list_element_size} bytes")

python_list = [1, 2, 3, 4, 5]
numpy_array = np.array([1, 2, 3, 4, 5])


start_time = time.time()
doubled_list = [x * 2 for x in python_list]
end_time = time.time()
print(f"Time taken for element-wise operation on Python list: {end_time - start_time} seconds")


start_time = time.time()
sum_numpy_array = np.sum(numpy_array)
end_time = time.time()
print(f"Time taken for summing NumPy array: {end_time - start_time} seconds")
