import numpy as np
import cv2
npy_path = "/Users/chenjingkun/Documents/code/image_tool/preprocessing/C0_gt_1.npy"

def showImage(image):
    cv2.imshow("image",image)
    cv2.waitKey(0)

data = np.load(npy_path)
print(data.shape)
for i in data:
    print(np.amax(i))
    showImage(i)
