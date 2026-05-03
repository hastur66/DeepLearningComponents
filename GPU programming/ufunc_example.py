from numba import vectorize
import numpy as np
import time


a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = np.arange(4*4).reshape((4,4))


@vectorize(["int64(int64, int64)"], target="cuda")
def add_ufunc(x, y):
    return x + y

# Time the numpy add function
start_time = time.time()
np.add(b, c)
end_time = time.time()
print(f"numpy add time: {end_time - start_time:.6f} seconds")

# Time the ufunc add_ufunc function
start_time = time.time()
add_ufunc(a, b)
end_time = time.time()
print(f"add_ufunc time: {end_time - start_time:.6f} seconds")
