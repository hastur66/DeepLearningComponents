from numba import cuda, float32
import numpy as np


# Threads per block must be a power of 2 for reduction (e.g., 128, 256)
THREADS_PER_BLOCK = 256


@cuda.jit
def mse_loss_kernel(predictions, targets, total_elements, partial_loss_sum):
    """
    Computes Mean Squared Error Loss: 1/N * sum((pred - target)^2)
    predictions:      (Batch_Size, Num_Outputs)
    targets:          (Batch_Size, Num_Outputs)
    total_elements:   Total count of float elements (Batch_Size * Num_Outputs)
    partial_loss_sum: Single-element array to store total accumulated squared error
    """
    # Allocate shared memory for threads within this block
    s_loss = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float32)

    # Global 1D thread ID
    idx = cuda.grid(1)
    # Local thread ID within the block
    tx = cuda.threadIdx.x

    # 1. Step 1: Calculate individual squared errors and load to Shared Memory
    if idx < total_elements:
        # Convert 1D thread index to 2D matrix coordinates dynamically
        row = idx // predictions.shape[1]
        col = idx % predictions.shape[1]
        
        diff = predictions[row, col] - targets[row, col]
        s_loss[tx] = diff * diff
    else:
        s_loss[tx] = 0.0

    # Wait for all threads in the block to finish writing to shared memory
    cuda.syncthreads()

    # 2. Step 2: In-place Parallel Reduction tree inside Shared Memory
    # Divide the block size by 2 at each step
    stride = THREADS_PER_BLOCK // 2
    while stride > 0:
        if tx < stride:
            s_loss[tx] += s_loss[tx + stride]
        
        # Sync after every step of the tree reduction
        cuda.syncthreads()
        stride //= 2

    # 3. Step 3: Thread 0 writes the block's sum atomically to Global Memory
    if tx == 0:
        # atomic.add prevents multiple blocks from overwriting each other simultaneously
        cuda.atomic.add(partial_loss_sum, 0, s_loss[0])


def main():
    batch_size = 64
    num_outputs = 10
    total_elements = batch_size * num_outputs

    # Create dummy network outputs and actual target labels
    np.random.seed(42)
    h_predictions = np.random.randn(batch_size, num_outputs).astype(np.float32)
    h_targets = np.random.randn(batch_size, num_outputs).astype(np.float32)
    
    # Initialize the output accumulator to zero
    h_partial_sum = np.zeros(1, dtype=np.float32)

    # Move arrays to GPU
    d_predictions = cuda.to_device(h_predictions)
    d_targets = cuda.to_device(h_targets)
    d_partial_sum = cuda.to_device(h_partial_sum)

    # Calculate execution configuration
    blocks_per_grid = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Launch Loss Kernel
    mse_loss_kernel[blocks_per_grid, THREADS_PER_BLOCK](
        d_predictions, d_targets, total_elements, d_partial_sum
    )

    # Bring total sum back to host
    total_squared_error = d_partial_sum.copy_to_host()[0]
    
    # Finalize the Mean Squared Error on CPU (Normalize by N)
    gpu_mse_loss = total_squared_error / total_elements

    # --- NumPy CPU verification ---
    cpu_mse_loss = np.mean((h_predictions - h_targets) ** 2)
    
    print(f"GPU Computed MSE Loss: {gpu_mse_loss:.6f}")
    print(f"CPU Verified MSE Loss: {cpu_mse_loss:.6f}")
    print(f"Difference:            {abs(gpu_mse_loss - cpu_mse_loss):.6e}")

if __name__ == "__main__":
    main()
