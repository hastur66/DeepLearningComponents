from numba import cuda, vectorize
import numpy as np
from math import exp

# this 
n = 1000000
greyscales = np.floor(np.random.uniform(0, 255, n).astype(np.float32))
weights = np.random.normal(.5, .1, n).astype(np.float32)


@vectorize(['float32(float32)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255


@vectorize(['float32(float32,float32)'], target='cuda')
def weigh(values, weights):
    return values * weights


@vectorize(['float32(float32)'], target='cuda')    
def activate(values):
    return ( exp(values) - exp(-values) ) / ( exp(values) + exp(-values) )


def create_hidden_layer(n, greyscales, weights, exp, normalize, weigh, activate):
    
    d_greyscales = cuda.to_device(greyscales)
    d_weights = cuda.to_device(weights)
    
    normalized = normalize(d_greyscales)
    weighted = weigh(normalized, d_weights)
    activated = activate(weighted)
    
    out_host = activated.copy_to_host()
    return activated


arguments = {"n":n,
            "greyscales": greyscales,
            "weights": weights,
            "exp": exp,
            "normalize": normalize,
            "weigh": weigh,
            "activate": activate}

a = create_hidden_layer(**arguments)
print(a)
print(a.copy_to_host())
