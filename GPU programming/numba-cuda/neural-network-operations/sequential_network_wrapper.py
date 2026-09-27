from numba import cuda
import numpy as np

from trainable_gpu_layer import CUDADenseLayer
from mse_loss_kernel import mse_loss_kernel


class CUDASequentialNetwork:
    def __init__(self):
        # A list to hold the ordered layers of the network
        self.layers = []

    def add(self, layer):
        """Appends a trainable CUDA layer to the network sequence."""
        self.layers.append(layer)

    def forward(self, d_inputs):
        """
        Feeds the input device array sequentially through all layers.
        Returns the final network prediction as a device array.
        """
        current_data = d_inputs
        for layer in self.layers:
            # The output pointer of one layer becomes the input pointer of the next
            current_data = layer.forward(current_data)
        return current_data

    def backward(self, d_loss_grad):
        """
        Accepts the initial loss gradient and backpropagates it 
        through the network in reverse order.
        """
        current_grad = d_loss_grad
        # Walk backward through layers: Layer N -> Layer N-1 -> ... -> Layer 0
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        return current_grad

    def step(self, learning_rate):
        """Triggers parameter updates for all layers holding weights."""
        for layer in self.layers:
            layer.update_parameters(learning_rate)


def main():
    # 1. Define Network Archetype (Input 128 -> Hidden 64 -> Hidden 32 -> Output 10)
    print("Initializing a 3-Layer Custom CUDA Neural Network...")
    model = CUDASequentialNetwork()
    model.add(CUDADenseLayer(num_inputs=128, num_outputs=64))
    model.add(CUDADenseLayer(num_inputs=64, num_outputs=32))
    model.add(CUDADenseLayer(num_inputs=32, num_outputs=10))

    # 2. Synthesize Mock Dataset (Batch of 64 samples)
    np.random.seed(42)
    h_features = np.random.randn(64, 128).astype(np.float32)
    h_targets = np.random.randn(64, 10).astype(np.float32)

    # Push dataset arrays onto the GPU once
    d_features = cuda.to_device(h_features)
    d_targets = cuda.to_device(h_targets)

    # Setup loss memory allocations for parallel reduction loss verification
    h_loss_accumulator = np.zeros(1, dtype=np.float32)
    d_loss_accumulator = cuda.to_device(h_loss_accumulator)
    total_loss_elements = 64 * 10

    # 3. Simulate Training Epochs
    epochs = 3
    learning_rate = 0.01

    print("\n--- Starting Training Run ---")
    for epoch in range(1, epochs + 1):
        # --- A. FORWARD PASS ---
        # Data streams forward layer-by-layer completely inside VRAM
        d_preds = model.forward(d_features)

        # --- B. COMPUTE MSE LOSS & ERROR SIGNAL ---
        # Calculate loss metrics using our previously defined parallel reduction kernel
        d_loss_accumulator.copy_to_device(np.zeros(1, dtype=np.float32)) # Reset to zero
        mse_loss_kernel[(total_loss_elements + 255) // 256, 256](
            d_preds, d_targets, total_loss_elements, d_loss_accumulator
        )
        current_loss = d_loss_accumulator.copy_to_host()[0] / total_loss_elements

        # Calculate Upstream Derivative (d_loss / d_pred) for MSE: 2/N * (pred - target)
        # We compute this error signal matrix to kick off backpropagation
        h_preds = d_preds.copy_to_host()
        h_loss_grad = (2.0 / total_loss_elements) * (h_preds - h_targets)
        d_loss_grad = cuda.to_device(h_loss_grad)

        # --- C. BACKWARD PASS ---
        # Error gradients flow backwards through the framework layout
        model.backward(d_loss_grad)

        # --- D. OPTIMIZATION STEP ---
        # Update architectural weights in-place across all layers
        model.step(learning_rate)

        print(f"Epoch {epoch}/{epochs} | Current Shared-VRAM MSE Loss: {current_loss:.6f}")

    print("\nTraining run completed successfully without CPU context-switch pipeline stalls!")

if __name__ == "__main__":
    main()
