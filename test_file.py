import time

import numpy as np

dim = 1000000
a = np.random.rand(dim)
b = np.random.rand(dim)

start = time.time()
r = np.dot(a, b)
end = time.time()
print(f"start: {start}, end: {end}")
print(f"duration: {(end - start) * 1000}")
