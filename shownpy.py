import numpy as np
from readimg import showImage

npy_file = "/Users/chenjingkun/Documents/code/texture/C0_gt.npy"
npy_list = np.load(npy_file)
for i in npy_list:
    showImage(i)