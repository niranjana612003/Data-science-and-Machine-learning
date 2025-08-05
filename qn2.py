import numpy as np


arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array using np.array():")
print(arr_1d)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array using np.array():")
print(arr_2d)

arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3D Array using np.array():")
print(arr_3d)


zeros_1d = np.zeros(5)
print("\n1D zeros:", zeros_1d)

zeros_2d = np.zeros((3, 4))
print("\n2D zeros:\n", zeros_2d)

zeros_3d = np.zeros((2, 3, 4))
print("\n3D zeros:\n", zeros_3d)


ones_1d = np.ones(5)
print("\n1D ones:", ones_1d)

ones_2d = np.ones((2, 3))
print("\n2D ones:\n", ones_2d)

ones_3d = np.ones((3, 2, 2))
print("\n3D ones:\n", ones_3d)



arange_1d = np.arange(10)
print("\n1D arange:", arange_1d)


arange_2d = np.arange(12).reshape(3, 4)
print("\n2D arange reshaped (3x4):\n", arange_2d)

arange_3d = np.arange(24).reshape(2, 3, 4)
print("\n3D arange reshaped (2x3x4):\n", arange_3d)


