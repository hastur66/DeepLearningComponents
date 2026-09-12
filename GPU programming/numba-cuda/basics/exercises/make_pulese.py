from numba import cuda, vectorize
import numpy as np
import math

import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


n = 100000
noise = (np.random.normal(size=n) * 3).astype(np.float32)
t = np.arange(n, dtype=np.float32)
period = n / 23

d_noise = cuda.to_device(noise)
d_t = cuda.to_device(t)

@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y


@vectorize(['float32(float32, float32, float32)'], target='cuda')
def make_pulses(i, period, amplitude):
    return max(math.sin(i / period) - 0.3, 0.0) * amplitude


d_pulses = make_pulses(d_t, period, 100.0)
waveform = add_ufunc(d_pulses, d_noise)

plt.plot(waveform.copy_to_host())
plt.show()
