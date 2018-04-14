import torch
import numpy as np
from torch import IntTensor, ByteTensor, CharTensor


x = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
x = torch.from_numpy(x)
print(x.shape)

y = np.array([-1, -2])
y = torch.from_numpy(y)
z = x * y
print(z.shape)

print('x', x)
print('y', y)
print('z', z)
