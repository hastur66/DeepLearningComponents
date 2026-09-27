from numba import cuda, float32
import numpy as np


@cuda.jit
def sgd_optimizer_kernel(weights, biases, d_weights, d_biases, learning_rate, weight_decay, num_outputs, num_inputs):
    """
    Updates weights and biases using Stochastic Gradient Descent (SGD) with Weight Decay.
    weights:       (Num_Outputs, Num_Inputs) -> Modified In-Place
    biases:        (Num_Outputs,)            -> Modified In-Place
    d_weights:     (Num_Outputs, Num_Inputs) -> Gradients from backward pass
    d_biases:      (Num_Outputs,)            -> Gradients from backward pass
    learning_rate: Step size multiplier (scalar float)
    weight_decay:  L2 Regularization strength (scalar float)
    """
    # Map threads to 2D coordinates (Output Neuron, Input Feature)
    out_idx, in_idx = cuda.grid(2)

    # 1. Update Weights Matrix
    if out_idx < num_outputs and in_idx < num_inputs:
        # Fetch current values
        w = weights[out_idx, in_idx]
        dw = d_weights[out_idx, in_idx]
        
        # Apply SGD formula with Weight Decay (L2 Regularization)
        weights[out_idx, in_idx] = w - (learning_rate * dw) - (learning_rate * weight_decay * w)
        
        # 2. Update Biases Vector (only handled by the first column of threads to avoid race conditions)
        if in_idx == 0:
            b = biases[out_idx]
            db = d_biases[out_idx]
            
            # Biases generally don't get weight decay applied
            biases[out_idx] = b - (learning_rate * db)


def main():
    # Dimensions matching our previous layers
    num_inputs = 128
    num_outputs = 32
    
    # Hyperparameters
    learning_rate = float32(0.01)
    weight_decay = float32(0.0001)

    # Initialize weights and mock gradients
    np.random.seed(42)
    h_weights = np.random.randn(num_outputs, num_inputs).astype(np.float32)
    h_biases = np.random.randn(num_outputs).astype(np.float32)
    
    # Dummy gradients calculated from a backward pass
    h_d_weights = np.random.randn(num_outputs, num_inputs).astype(np.float32)
    h_d_biases = np.random.randn(num_outputs).astype(np.float32)

    # Clone for CPU verification comparison
    h_weights_cpu = h_weights.copy()
    h_biases_cpu = h_biases.copy()

    # Move parameters to GPU (Note: we use to_device because we modify them in-place)
    d_weights = cuda.to_device(h_weights)
    d_biases = cuda.to_device(h_biases)
    d_d_weights = cuda.to_device(h_d_weights)
    d_d_biases = cuda.to_device(h_d_biases)

    # Configure 2D thread grid configuration
    threads_per_block = (16, 16)
    grid_x = (num_outputs + 15) // 16
    grid_y = (num_inputs + 15) // 16

    # Launch Optimizer Kernel
    sgd_optimizer_kernel[(grid_x, grid_y), threads_per_block](
        d_weights, d_biases, d_d_weights, d_d_biases, 
        learning_rate, weight_decay, num_outputs, num_inputs
    )

    # Retrieve updated values back to Host
    updated_weights_gpu = d_weights.copy_to_host()
    updated_biases_gpu = d_biases.copy_to_host()

    # --- NumPy CPU Verification ---
    updated_weights_cpu = h_weights_cpu - (learning_rate * h_d_weights) - (learning_rate * weight_decay * h_weights_cpu)
    updated_biases_cpu = h_biases_cpu - (learning_rate * h_d_biases)

    # Verify accuracy
    weight_diff = np.max(np.abs(updated_weights_gpu - updated_weights_cpu))
    bias_diff = np.max(np.abs(updated_biases_gpu - updated_biases_cpu))
    
    print("Optimizer update step complete!")
    print(f"Max Weight Difference vs CPU: {weight_diff:.6e}")
    print(f"Max Bias Difference vs CPU:   {bias_diff:.6e}")
    # Expected output: differences are NOT exactly zero due to float32 rounding.
    # GPU and CPU may reorder floating point operations slightly differently,
    # causing last-bit rounding divergence (~1e-7 for weights, ~1e-9 for biases).
    # This is within float32 machine epsilon (1.19e-07) and is numerically harmless.
    # Bias difference is smaller because its formula (b - lr*db) has fewer operations.

if __name__ == "__main__":
    main()
