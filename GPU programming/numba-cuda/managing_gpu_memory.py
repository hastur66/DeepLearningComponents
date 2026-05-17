from numba import cuda, vectorize
import numpy as np
import time


@vectorize(["float32(float32, float32)"], target="cuda")
def add_ufunc(a, b):
    return a + b

n = 1000000
x = np.arange(n).astype(np.float32)
y = 2 * x


start_time = time.time()
add_ufunc(x, y)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.6f} seconds")


x_device = cuda.to_device(x)
y_device = cuda.to_device(y)

print(x_device)
print(x_device.shape)
print(x_device.dtype)

start_time = time.time()
add_ufunc(x_device, y_device)
end_time = time.time()

print(f"Execution time on GPU: {end_time - start_time:.6f} seconds")


out_device = cuda.device_array(shape=(n,), dtype=np.float32)

start_time = time.time()
add_ufunc(x_device, y_device, out=out_device)
end_time = time.time()

print(f"Execution time with preallocated device array: {end_time - start_time:.6f} seconds")

out_host = out_device.copy_to_host()
print(out_host[:5])
