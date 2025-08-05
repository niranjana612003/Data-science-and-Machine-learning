import numpy as np

A = np.array([[2, 1],
              [3, 2]])
b = np.array([8, 13])

# Solve for x (which contains [x, y])
solution = np.linalg.solve(A, b)
print("Solution (x, y):", solution)
