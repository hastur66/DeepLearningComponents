from numba import cuda
import numpy as np


@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    out[idx] = x[idx] + y[idx]

n = 4096
x = np.arange(n).astype(np.int32)
y = np.ones_like(x)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(d_x)

threads_per_block = 128
blocks_per_grid = n // threads_per_block

add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
cuda.synchronize()
result = d_out.copy_to_host()
print(result)
