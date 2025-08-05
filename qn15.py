import numpy as np

A = np.array([[4, 7],
              [2, 6]])
print("Matrix A:\n", A)

det_A = np.linalg.det(A)
print("\nDeterminant of A:", det_A)


inv_A = np.linalg.inv(A)
print("\nInverse of A:\n", inv_A)


identity = np.dot(A, inv_A)
print("\nProduct of A and its inverse (should be identity):\n", identity)
