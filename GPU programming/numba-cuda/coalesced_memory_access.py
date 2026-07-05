from numba import cuda
import numpy as np
import timeit


n = 1024*12024

thread_per_block = 1024
blocks = int(n / thread_per_block)

stride = 16


a = np.ones(stride * n).astype(np.float32)
b = a.copy().astype(np.float32)

out = np.zeros(n).astype(np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_out = cuda.to_device(out)


@cuda.jit
def add_experiment(a, b, out, stride, coalesced):
    i = cuda.grid(1)

    if coalesced == True:
        out[i] = a[i] + b[i]
    else:
        out[i] = a[stride*i] + b[stride*i]


# warmup
add_experiment[blocks, thread_per_block](d_a, d_b, d_out, stride, True)
add_experiment[blocks, thread_per_block](d_a, d_b, d_out, stride, False)
cuda.synchronize()

# benchmark
N = 100
coalesced = timeit.timeit(lambda: (add_experiment[blocks, thread_per_block](d_a, d_b, d_out, stride, True), cuda.synchronize()), number=N) / N
uncoalesced = timeit.timeit(lambda: (add_experiment[blocks, thread_per_block](d_a, d_b, d_out, stride, False), cuda.synchronize()), number=N) / N

print(f"coalesced:   {coalesced:.6f}s")
print(f"uncoalesced: {uncoalesced:.6f}s")
