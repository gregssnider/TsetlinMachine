import torch
from torch import IntTensor, ByteTensor, CharTensor

x = IntTensor(4, 5)
y = IntTensor(4, 5)
compare = x > y
print(type(compare))