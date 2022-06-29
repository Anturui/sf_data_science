"""Игра угадай число.
Компьютер сам загадывает и угадывает число
Алгоритм модифицирован, чтобы компьютер отгадывал число за менее чем 20 попыток 
"""
import numpy as np
print( -(2**16)/2,(2**16)/2 - 1)
a = np.int8(25)
print(a)
print(type(a))
print(np.iinfo(np.int8))
print(np.iinfo(np.int16))
print(np.iinfo(np.int32))
print(np.iinfo(np.int64))

b = np.uint8(124)
print(b)
# 124
print(type(b))
# <class 'numpy.uint8'>
print(np.iinfo(b))
# iinfo(min=0, max=255, dtype=uint8)
print(2*((2**128)/2 - 1))

print(*sorted(map(str, set(np.sctypeDict.values()))), sep='\n')

a = True
print(type(a))
# <class 'bool'>
a = np.bool(a)
print(type(a))
# <class 'bool'>
a = np.bool_(a)
print(type(a))
# <class 'numpy.bool_'>
 
# Значения равны
print(np.bool(True) == np.bool_(True))
# True
# А типы — нет:
print(type(np.bool(True)) == type(np.bool_(True)))
# False

arr = np.array([1,5,2,9,10])
print(arr.dtype)