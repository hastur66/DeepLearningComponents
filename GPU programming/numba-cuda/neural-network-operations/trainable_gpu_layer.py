from numba import cuda, float32
import numpy as np


# Configuration matching our optimized setup
TILE_SIZE = 16

# ==========================================
# 1. CUDA KERNELS DEFINITIONS
# ==========================================

@cuda.jit
def _forward_kernel(inputs, weights, bias, output):
    """Batched and Tiled Forward Pass with ReLU."""
    s_inputs = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    s_weights = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

    row, col = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    batch_size, num_inputs = inputs.shape
    num_outputs = weights.shape[0]
    tmp = float32(0.0)
    num_tiles = (num_inputs + TILE_SIZE - 1) // TILE_SIZE
    
    for t in range(num_tiles):
        if row < batch_size and (t * TILE_SIZE + ty) < num_inputs:
            s_inputs[tx, ty] = inputs[row, t * TILE_SIZE + ty]
        else:
            s_inputs[tx, ty] = 0.0

        if col < num_outputs and (t * TILE_SIZE + tx) < num_inputs:
            s_weights[tx, ty] = weights[col, t * TILE_SIZE + tx]
        else:
            s_weights[tx, ty] = 0.0

        cuda.syncthreads()

        for k in range(TILE_SIZE):
            tmp += s_inputs[tx, k] * s_weights[k, ty]

        cuda.syncthreads()

    if row < batch_size and col < num_outputs:
        val = tmp + bias[col]
        output[row, col] = val if val > 0.0 else 0.0

@cuda.jit
def _backward_weights_kernel(inputs, d_loss_d_output, forward_outputs, d_weights, d_bias):
    """Computes gradients for weights and biases."""
    out_idx, in_idx = cuda.grid(2)
    num_outputs, num_inputs = d_weights.shape
    batch_size = inputs.shape[0]

    if out_idx < num_outputs and in_idx < num_inputs:
        weight_grad_sum = float32(0.0)
        bias_grad_sum = float32(0.0)
        
        for b in range(batch_size):
            if forward_outputs[b, out_idx] > 0.0:
                upstream_grad = d_loss_d_output[b, out_idx]
                weight_grad_sum += inputs[b, in_idx] * upstream_grad
                if in_idx == 0:
                    bias_grad_sum += upstream_grad
        
        d_weights[out_idx, in_idx] = weight_grad_sum / batch_size # Normalized by batch size
        if in_idx == 0:
            d_bias[out_idx] = bias_grad_sum / batch_size

@cuda.jit
def _backward_inputs_kernel(d_loss_d_output, weights, forward_outputs, d_inputs):
    """Computes gradients for inputs to pass to the previous layer."""
    batch_idx, in_idx = cuda.grid(2)
    batch_size, num_inputs = d_inputs.shape
    num_outputs = weights.shape[0]

    if batch_idx < batch_size and in_idx < num_inputs:
        input_grad_sum = float32(0.0)
        for o in range(num_outputs):
            if forward_outputs[batch_idx, o] > 0.0:
                input_grad_sum += d_loss_d_output[batch_idx, o] * weights[o, in_idx]
                
        d_inputs[batch_idx, in_idx] = input_grad_sum

@cuda.jit
def _sgd_step_kernel(weights, biases, d_weights, d_biases, learning_rate, num_outputs, num_inputs):
    """Applies weight updates."""
    out_idx, in_idx = cuda.grid(2)
    if out_idx < num_outputs and in_idx < num_inputs:
        weights[out_idx, in_idx] -= learning_rate * d_weights[out_idx, in_idx]
        if in_idx == 0:
            biases[out_idx] -= learning_rate * d_biases[out_idx]

# ==========================================
# 2. THE CUDADENSELAYER CLASS
# ==========================================

class CUDADenseLayer:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Initialize parameters on Host (He/Xavier initialization style)
        h_weights = (np.random.randn(num_outputs, num_inputs) * np.sqrt(2.0 / num_inputs)).astype(np.float32)
        h_biases = np.zeros(num_outputs, dtype=np.float32)

        # Allocate and hold permanently on GPU Device Memory
        self.d_weights = cuda.to_device(h_weights)
        self.d_biases = cuda.to_device(h_biases)
        self.d_d_weights = cuda.device_array((num_outputs, num_inputs), dtype=np.float32)
        self.d_d_biases = cuda.device_array(num_outputs, dtype=np.float32)

        # Cache fields for data shapes dynamically allocated during forward steps
        self.d_inputs = None
        self.d_outputs = None

    def forward(self, d_inputs):
        """Accepts device array inputs, executes forward pass, returns device array outputs."""
        self.d_inputs = d_inputs # Keep pointer to input for backward pass
        batch_size = d_inputs.shape[0]

        # Allocate output buffer on GPU
        self.d_outputs = cuda.device_array((batch_size, self.num_outputs), dtype=np.float32)

        # Configure Grid
        threads = (TILE_SIZE, TILE_SIZE)
        grid = ((batch_size + TILE_SIZE - 1) // TILE_SIZE, (self.num_outputs + TILE_SIZE - 1) // TILE_SIZE)

        _forward_kernel[grid, threads](self.d_inputs, self.d_weights, self.d_biases, self.d_outputs)
        return self.d_outputs

    def backward(self, d_loss_d_output):
        """Computes gradients and returns d_loss_d_input to pass backward to previous layers."""
        batch_size = self.d_inputs.shape[0]
        d_d_inputs = cuda.device_array((batch_size, self.num_inputs), dtype=np.float32)

        threads = (16, 16)
        
        # 1. Grid configurations
        grid_w = ((self.num_outputs + 15) // 16, (self.num_inputs + 15) // 16)
        grid_in = ((batch_size + 15) // 16, (self.num_inputs + 15) // 16)

        # 2. Launch Gradients calculation
        _backward_weights_kernel[grid_w, threads](
            self.d_inputs, d_loss_d_output, self.d_outputs, self.d_d_weights, self.d_d_biases
        )
        _backward_inputs_kernel[grid_in, threads](
            d_loss_d_output, self.d_weights, self.outputs if hasattr(self, 'outputs') else self.d_outputs, d_d_inputs
        )

        return d_d_inputs

    def update_parameters(self, learning_rate):
        """Executes an in-place optimization step on the GPU weights."""
        threads = (16, 16)
        grid = ((self.num_outputs + 15) // 16, (self.num_inputs + 15) // 16)

        _sgd_step_kernel[grid, threads](
            self.d_weights, self.d_biases, self.d_d_weights, self.d_d_biases, 
            np.float32(learning_rate), self.num_outputs, self.num_inputs
        )


if __name__ == "__main__":
    # 1. Instantiate the layer (128 inputs -> 32 outputs)
    layer = CUDADenseLayer(num_inputs=128, num_outputs=32)
    
    # 2. Create mock training input batches (e.g., Batch Size of 64)
    np.random.seed(42)
    h_batch_inputs = np.random.randn(64, 128).astype(np.float32)
    d_batch_inputs = cuda.to_device(h_batch_inputs)

    # 3. RUN FORWARD PASS
    print("Executing Forward Pass...")
    d_predictions = layer.forward(d_batch_inputs)
    
    # 4. Generate a mock target gradient (Upstream loss derivative)
    h_mock_upstream_grad = np.random.randn(64, 32).astype(np.float32)
    d_mock_upstream_grad = cuda.to_device(h_mock_upstream_grad)

    # 5. RUN BACKWARD PASS
    print("Executing Backward Pass...")
    d_inputs_grad = layer.backward(d_mock_upstream_grad)

    # Record weight values before optimizing
    weights_before = layer.d_weights.copy_to_host()

    # 6. RUN OPTIMIZER UPDATE
    print("Updating weights via SGD Kernel...")
    layer.update_parameters(learning_rate=0.05)

    # Record weight values after optimizing
    weights_after = layer.d_weights.copy_to_host()

    # 7. Verification check
    assert not np.array_equal(weights_before, weights_after), "Weights failed to update!"
    print("\nLayer structural loop test completed successfully!")
    print(f"Predictions Shape:      {d_predictions.shape}")
    print(f"Input Gradients Shape:  {d_inputs_grad.shape}")
    print(f"Average absolute weight change: {np.mean(np.abs(weights_after - weights_before)):.6e}")
