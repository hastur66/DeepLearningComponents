from numba import cuda
import numpy as np


# This is a race condition
@cuda.jit
def thread_counter_race_condition(global_counter):
    global_counter[0] += 1


# This is safe
@cuda.jit
def thread_counter_safe(global_counter):
    cuda.atomic.add(global_counter, 0, 1) 


# Race condition example
global_counter = cuda.to_device(np.array([0], dtype=np.int32))
thread_counter_race_condition[64, 64](global_counter)

print('Should be %d:' % (64*64), global_counter.copy_to_host())


# Now with atomic operations
global_counter = cuda.to_device(np.array([0], dtype=np.int32))
thread_counter_safe[64, 64](global_counter)

print('Should be %d:' % (64*64), global_counter.copy_to_host())
