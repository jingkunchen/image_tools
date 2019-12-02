import numpy as np
from readimg import showImage


npy_path = "/Users/chenjingkun/Documents/data/stacom/npy/lge_data.npy"

data = np.load(npy_path)
print(data)
for i in data:
    showImage(i)