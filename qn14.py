import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

result_dot = np.dot(A, B)
print("Result using np.dot():\n", result_dot)

result_at = A @ B
print("\nResult using @ operator:\n", result_at)

