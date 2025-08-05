import numpy as np
import sys
import time

size = 1_000_000
py_list = list(range(size))
np_array = np.arange(size)

print("Memory used by Python list (in bytes):", sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list))
print("Memory used by NumPy array (in bytes):", np_array.nbytes)


start_list = time.time()
py_result = [x + 1 for x in py_list]
end_list = time.time()

start_np = time.time()
np_result = np_array + 1
end_np = time.time()

print(f"\nTime for list addition: {end_list - start_list:.5f} seconds")
print(f"Time for NumPy array addition: {end_np - start_np:.5f} seconds")

