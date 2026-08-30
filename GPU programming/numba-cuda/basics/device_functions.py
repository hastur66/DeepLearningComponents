from numba import cuda, vectorize
import math
import numpy as np

@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x  = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y


@vectorize(["float32(float32, float32, float32, float32)"], target="cuda")
def polar_distance(rho_1, theta_1, rho_2, theta_2):
    x1, y1 = polar_to_cartesian(rho_1, theta_1)
    x2, y2 = polar_to_cartesian(rho_2, theta_2)
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


n = 1000000
rho_1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

polar_distance(rho_1, theta1, rho2, theta2)