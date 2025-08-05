import numpy as np

data = list(range(10))

squared_loop = []
for x in data:
    squared_loop.append(x ** 2)

print("Squared using loop:", squared_loop)


data_np = np.arange(10)

squared_vectorized = data_np ** 2

print("Squared using NumPy vectorization:", squared_vectorized)
