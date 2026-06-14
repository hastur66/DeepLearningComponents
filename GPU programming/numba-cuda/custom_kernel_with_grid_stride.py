from numba import cuda
import numpy as np


@cuda.jit
def add_kernel(x, y, out):
    start =  cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.size, stride):
        out[i] = x[i] + y[i]


n = 100000 # more elements than threads in our grid
x = np.arange(n).astype(np.int32)
y = np.ones_like(x)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(d_x)

threads_per_block = 128
blocks_per_grid = 256

add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
print(d_out.copy_to_host())
