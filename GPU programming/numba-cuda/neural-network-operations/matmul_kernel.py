from numba import cuda
import numpy as np


@cuda.jit
def matmul_kernel(A, B, C):
    # Get the row and column coordinates in the 2D grid
    row, col = cuda.grid(2)
    
    # Boundary check for the output dimensions
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        # Dot product of row A and column B
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


# Example Usage: Batch of 128 items, 512 input features, 256 output neurons
M, K, N = 128, 512, 256
A_host = np.random.randn(M, K).astype(np.float32)
B_host = np.random.randn(K, N).astype(np.float32)
C_device = cuda.device_array((M, N), dtype=np.float32)

# Copy to device
A_device = cuda.to_device(A_host)
B_device = cuda.to_device(B_host)

# 2D Grid configuration (using 16x16 thread blocks)
threads_per_block = (16, 16)
blocks_x = (C_device.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (C_device.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_x, blocks_y)

matmul_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)

print(C_device.copy_to_host())
