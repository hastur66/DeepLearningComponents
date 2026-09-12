from numba import cuda
import numpy as np


# 1. Define a basic element-wise kernel
@cuda.jit
def process_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2  # Example operation

# 2. Setup data dimensions and streaming parameters
data_size = 100_000_000
chunk_size = 10_000_000
num_chunks = data_size // chunk_size

# 3. CRITICAL: Allocate Pinned Host Memory for asynchronous transfers
# This allocates memory on the CPU that the GPU can access directly via DMA.
h_data = cuda.pinned_array(data_size, dtype=np.float32)
h_data[:] = np.arange(data_size, dtype=np.float32) # Initialize data
h_result = cuda.pinned_array(data_size, dtype=np.float32)

# 4. Create CUDA Streams
stream1 = cuda.stream()
stream2 = cuda.stream()
streams = [stream1, stream2]

# 5. Pipeline Execution Loop
threads_per_block = 256
blocks_per_grid = (chunk_size + (threads_per_block - 1)) // threads_per_block

for i in range(num_chunks):
    # Alternately pick stream 1 or 2
    stream = streams[i % 2]
    
    # Calculate array offsets for the current chunk
    start = i * chunk_size
    end = start + chunk_size
    
    # Allocate device array bound to the specific stream
    d_chunk = cuda.to_device(h_data[start:end], stream=stream)
    
    # Launch the kernel asynchronously inside the stream
    process_kernel[blocks_per_grid, threads_per_block, stream](d_chunk)
    
    # Copy results back to the pinned host memory asynchronously
    d_chunk.copy_to_host(h_result[start:end], stream=stream)

# 6. Synchronize the GPU to ensure all streams have finished before reading results
cuda.synchronize()

# Verify a small slice of the result
print("Success!" if np.allclose(h_result[:10], np.arange(10) * 2) else "Failed!")
