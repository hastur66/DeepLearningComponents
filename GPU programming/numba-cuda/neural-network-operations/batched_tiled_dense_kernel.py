from numba import cuda, float32
import numpy as np


# Define the tile size (must match block dimensions)
TILE_SIZE = 16


@cuda.jit
def batched_tiled_dense_kernel(inputs, weights, bias, output):
    """
    Inputs shape:  (Batch_Size, Num_Inputs)
    Weights shape: (Num_Outputs, Num_Inputs)  -- Note: storing as (Out, In) for dot product
    Bias shape:    (Num_Outputs,)
    Output shape:  (Batch_Size, Num_Outputs)
    """
    # 1. Allocate shared memory tiles for the current block
    # Static shared memory allocation requires compile-time constants
    s_inputs = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    s_weights = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

    # Absolute 2D matrix coordinates for this thread
    row, col = cuda.grid(2)  # row -> Batch Index, col -> Output Neuron Index

    # Relative coordinates within the local thread block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Retrieve matrix bounds dynamically
    batch_size, num_inputs = inputs.shape
    num_outputs = weights.shape[0]

    # Accumulator for the dot product
    tmp = float32(0.0)

    # 2. Loop over all tiles required to cover 'num_inputs'
    num_tiles = (num_inputs + TILE_SIZE - 1) // TILE_SIZE
    
    for t in range(num_tiles):
        # Collaborative Load: Thread loads one element into the input tile
        if row < batch_size and (t * TILE_SIZE + ty) < num_inputs:
            s_inputs[tx, ty] = inputs[row, t * TILE_SIZE + ty]
        else:
            s_inputs[tx, ty] = 0.0

        # Collaborative Load: Thread loads one element into the weight tile
        if col < num_outputs and (t * TILE_SIZE + tx) < num_inputs:
            s_weights[tx, ty] = weights[col, t * TILE_SIZE + tx]
        else:
            s_weights[tx, ty] = 0.0

        # Synchronize to make sure all threads finished loading the tile
        cuda.syncthreads()

        # 3. Compute partial dot product from the cached shared memory tile
        for k in range(TILE_SIZE):
            tmp += s_inputs[tx, k] * s_weights[k, ty]

        # Synchronize before loading the next tile to avoid overwriting active data
        cuda.syncthreads()

    # 4. Write final result to global memory if within bounds
    if row < batch_size and col < num_outputs:
        # Add bias and apply ReLU activation inline
        val = tmp + bias[col]
        output[row, col] = val if val > 0.0 else 0.0


# --- Host Execution Wrapper ---
def main():
    # Parameters
    batch_size = 64
    num_inputs = 128
    num_outputs = 32

    # Initialize mock data
    np.random.seed(42)
    h_inputs = np.random.randn(batch_size, num_inputs).astype(np.float32)
    h_weights = np.random.randn(num_outputs, num_inputs).astype(np.float32)
    h_bias = np.random.randn(num_outputs).astype(np.float32)
    h_output = np.zeros((batch_size, num_outputs), dtype=np.float32)

    # Copy to GPU device
    d_inputs = cuda.to_device(h_inputs)
    d_weights = cuda.to_device(h_weights)
    d_bias = cuda.to_device(h_bias)
    d_output = cuda.device_array_like(h_output)

    # Execution configuration (2D block and grid matching TILE_SIZE)
    threads_per_block = (TILE_SIZE, TILE_SIZE)
    blocks_per_grid_x = (batch_size + TILE_SIZE - 1) // TILE_SIZE
    blocks_per_grid_y = (num_outputs + TILE_SIZE - 1) // TILE_SIZE
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    batched_tiled_dense_kernel[blocks_per_grid, threads_per_block](
        d_inputs, d_weights, d_bias, d_output
    )

    # Copy back and verify shape
    h_output = d_output.copy_to_host()
    print("Execution complete!")
    print(f"Input Shape:  {h_inputs.shape}")
    print(f"Output Shape: {h_output.shape} (Successfully batched)")
    print(f"Sample predictions (first 2 rows):\n{h_output[:2, :5]}")

if __name__ == "__main__":
    main()
