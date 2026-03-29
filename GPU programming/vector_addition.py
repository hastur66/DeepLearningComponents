import numpy as np
from numba import cuda


@cuda.jit
def vector_add_kernel(a, b, result):

    pos = cuda.grid(1)

    if pos < a.size:
        result[pos] = a[pos] + b[pos]


n = 1000000
a = np.array(np.random.random(n), dtype=np.float32)
b = np.array(np.random.random(n), dtype=np.float32)
result = np.zeros(n, dtype=np.float32)


d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_result = cuda.device_array_like(result)


threads_per_block = 256
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_result)

result = d_result.copy_to_host()
print(f"Result (first 5): {result[:5]}")
