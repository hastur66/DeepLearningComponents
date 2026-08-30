from numba import cuda, types
import numpy as np


@cuda.jit
def swap_with_shared(vector, swapped):
    # shared memory array to store vector elements
    temp = cuda.shared.array(4, dtype=types.int32)

    idx = cuda.grid(1)

    # Move the vector elements from global memory  to shared memory
    temp[idx] = vector[idx]

    # Wait until all the threads in the block have written their values
    cuda.syncthreads()
    
    # Copy from shared memory to global memory in reversed order
    swapped[idx] = temp[3 - cuda.threadIdx.x]


vector = np.arange(4).astype(np.int32)
swapped = np.zeros_like(vector)

d_vector = cuda.to_device(vector)
d_swapped = cuda.to_device(swapped)

swap_with_shared[1, 4](d_vector, d_swapped)

result = d_swapped.copy_to_host()
result
