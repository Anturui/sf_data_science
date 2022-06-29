import numpy as np
arr = np.array([1,5,2,9,10])
print(arr.dtype)
arr = np.array([1,5,2,9,10], dtype=np.int8)
print(arr)
arr[2] = 2000
print(arr)
arr = np.array([1,5,2,9,10], dtype=np.int8)
nd_arr = np.array([
               [12, 45, 78],
               [34, 56, 13],
               [12, 98, 76]
               ], dtype=np.int16)
print(arr.ndim)
print(nd_arr.ndim)

print(arr.size)
print(nd_arr.size)

print(arr.shape)
print(nd_arr.shape)

print(arr.itemsize)
print(nd_arr.itemsize)

zeros_1d = np.zeros(5)
print(zeros_1d)

zeros_3d = np.zeros((5,4,3), dtype=np.float32)
print(zeros_3d.shape)
# (5, 4, 3)

print( np.arange(2.5, 5, 0.5)) 
print(np.arange(2.5, 5, 0.5, dtype=np.float16)) 

print(np.linspace(1, 2, 10))
print(np.linspace(1, 2, 10, endpoint=False))
arr, step = np.linspace(1, 2, 10, endpoint=True, retstep=True)

print(step)

arr, step = np.linspace(1, 2, 10, endpoint=False, retstep=True)
print(step)

arr, step = np.linspace(-6, 21, 60, endpoint=False, retstep=True)
print(step)