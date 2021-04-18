import numpy as np

array = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
print(array)

array[0:2, 1:3] = np.array([[7, 8],[9, 10]], dtype=np.int32)
print(array)

array = np.array([1,2,3], dtype=np.int32)
print(array>=2)

array_filtered = np.delete(array, array>=2)
print(array_filtered)

array = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
print(array>=3)

array_filtered = np.delete(array, [1], axis=0)
print(array_filtered)
