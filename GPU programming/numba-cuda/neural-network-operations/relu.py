from numba import cuda
import numpy as np


@cuda.jit
def relu_kernel(x, out):
    i = cuda.grid(1)
    if i < x.size:
        val = x[i]
        out[i] = val if val > 0 else 0.0


n = 10_000_000
x_host = np.random.randn(n).astype(np.float32)
out_host = np.empty_like(x_host)

# Transfer data to device (GPU)
x_device = cuda.to_device(x_host)
out_device = cuda.device_array_like(x_host)

# Configure threads and blocks
threads_per_block = 256
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

# Launch kernel
relu_kernel[blocks_per_grid, threads_per_block](x_device, out_device)

# Copy result back
out_host = out_device.copy_to_host()

print(out_host)
