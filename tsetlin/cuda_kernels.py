"""
CUDA kernels for the Tsetlin Machine. These will be replaced by Tensor
Comprehensions once those support bit-wise logical operations.

This is based on code from szagoruyk0:
    https://github.com/szagoruyko/pyinn/blob/master/pyinn/ncrelu.py
"""
from collections import namedtuple
import os
print(os.environ["LD_LIBRARY_PATH"])
import cupy
import torch

Stream = namedtuple('Stream', ['ptr'])

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code):
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


kernels = '''
extern "C"
__global__ void increment(int *output, int *input, int elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > elements)
        return;
    output[i] = input[i] + 1; 
}
'''

def do_increment(input):
    if not input.is_cuda:
        return input + 1
    assert input.is_contiguous()
    with torch.cuda.device_of(input):
        length = input.size()
        output = input.new(length)
        func = load_kernel('increment', kernels)
        func(args=[output.data_ptr(), input.data_ptr(), input.numel()],
             block=(CUDA_NUM_THREADS, 1, 1),
             grid=(GET_BLOCKS(input.numel()), 1, 1),
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return output


from time import time
tensor = torch.IntTensor([0, 1, 2, 3]).cuda()
# Warm up GPU caches
tensor = do_increment(tensor)
start_time = time()
for i in range(30000):
    tensor = do_increment(tensor)
elapsed_time = time() - start_time
print(tensor)
print('time:', elapsed_time)
