from numba import cuda, float32
import numpy as np


# 1. Kernel to calculate Gradients for Weights and Biases
@cuda.jit
def dense_backward_weights_kernel(inputs, d_loss_d_output, forward_outputs, d_weights, d_bias):
    """
    Computes gradients for weights and biases.
    inputs:          (Batch_Size, Num_Inputs)
    d_loss_d_output: (Batch_Size, Num_Outputs)
    forward_outputs: (Batch_Size, Num_Outputs)  -- Used to check ReLU state
    d_weights:       (Num_Outputs, Num_Inputs)  -- Output Gradient
    d_bias:          (Num_Outputs,)             -- Output Gradient
    """
    # Map threads to (Output Neuron, Input Feature)
    out_idx, in_idx = cuda.grid(2)
    
    num_outputs, num_inputs = d_weights.shape
    batch_size = inputs.shape[0]

    if out_idx < num_outputs and in_idx < num_inputs:
        weight_grad_sum = float32(0.0)
        bias_grad_sum = float32(0.0)
        
        # Loop over the entire batch to accumulate gradients
        for b in range(batch_size):
            # 1. ReLU Backward: check if the forward propagation pass was active (> 0)
            if forward_outputs[b, out_idx] > 0.0:
                upstream_grad = d_loss_d_output[b, out_idx]
                
                # 2. Accumulate weight gradient contribution from this batch item
                weight_grad_sum += inputs[b, in_idx] * upstream_grad
                
                # 3. Accumulate bias gradient (only once per output neuron thread column)
                if in_idx == 0:
                    bias_grad_sum += upstream_grad
        
        # Store the total averaged or accumulated gradient
        d_weights[out_idx, in_idx] = weight_grad_sum
        if in_idx == 0:
            d_bias[out_idx] = bias_grad_sum


# 2. Kernel to calculate Gradients for the Inputs (to pass to the previous layer)
@cuda.jit
def dense_backward_inputs_kernel(d_loss_d_output, weights, forward_outputs, d_inputs):
    """
    Computes gradients for inputs to pass backward.
    d_loss_d_output: (Batch_Size, Num_Outputs)
    weights:         (Num_Outputs, Num_Inputs)
    forward_outputs: (Batch_Size, Num_Outputs)
    d_inputs:        (Batch_Size, Num_Inputs)   -- Output Gradient
    """
    # Map threads to (Batch Index, Input Feature)
    batch_idx, in_idx = cuda.grid(2)
    
    batch_size, num_inputs = d_inputs.shape
    num_outputs = weights.shape[0]

    if batch_idx < batch_size and in_idx < num_inputs:
        input_grad_sum = float32(0.0)
        
        # Loop across all output neurons to see how much this input contributed
        for o in range(num_outputs):
            if forward_outputs[batch_idx, o] > 0.0:
                upstream_grad = d_loss_d_output[batch_idx, o]
                input_grad_sum += upstream_grad * weights[o, in_idx]
                
        d_inputs[batch_idx, in_idx] = input_grad_sum


def main():
    batch_size = 64
    num_inputs = 128
    num_outputs = 32

    # Synthesize dummy data representing a forward pass state
    h_inputs = np.random.randn(batch_size, num_inputs).astype(np.float32)
    h_weights = np.random.randn(num_outputs, num_inputs).astype(np.float32)
    h_forward_outputs = np.random.randn(batch_size, num_outputs).astype(np.float32)
    # Ensure some outputs are 0 to simulate ReLU clipping
    h_forward_outputs[h_forward_outputs < 0] = 0.0 
    
    # Upstream error gradient coming from the loss function/next layer
    h_d_loss_d_output = np.random.randn(batch_size, num_outputs).astype(np.float32)

    # Allocate space for output gradients
    h_d_weights = np.zeros_like(h_weights)
    h_d_bias = np.zeros(num_outputs, dtype=np.float32)
    h_d_inputs = np.zeros_like(h_inputs)

    # Move to GPU
    d_inputs = cuda.to_device(h_inputs)
    d_weights = cuda.to_device(h_weights)
    d_forward_outputs = cuda.to_device(h_forward_outputs)
    d_loss_d_output = cuda.to_device(h_d_loss_d_output)
    
    d_d_weights = cuda.device_array_like(h_d_weights)
    d_d_bias = cuda.device_array_like(h_d_bias)
    d_d_inputs = cuda.device_array_like(h_d_inputs)

    # --- Launch Kernel 1: Weight & Bias Gradients ---
    threads_per_block_1 = (16, 16)
    grid_x_1 = (num_outputs + 15) // 16
    grid_y_1 = (num_inputs + 15) // 16
    
    dense_backward_weights_kernel[(grid_x_1, grid_y_1), threads_per_block_1](
        d_inputs, d_loss_d_output, d_forward_outputs, d_d_weights, d_d_bias
    )

    # --- Launch Kernel 2: Input Gradients ---
    threads_per_block_2 = (16, 16)
    grid_x_2 = (batch_size + 15) // 16
    grid_y_2 = (num_inputs + 15) // 16
    
    dense_backward_inputs_kernel[(grid_x_2, grid_y_2), threads_per_block_2](
        d_loss_d_output, d_weights, d_forward_outputs, d_d_inputs
    )

    # Copy results back to verify execution
    grad_w = d_d_weights.copy_to_host()
    grad_in = d_d_inputs.copy_to_host()
    
    print("Backward pass successfully executed!")
    print("Weight Gradients Shape:", grad_w.shape)
    print("Input Gradients Shape:", grad_in.shape)

if __name__ == "__main__":
    main()
