import numpy as np
import cv2
lesion = np.load("resize/gt_array_c0.npy")
lesion[lesion == 1] = 500
lesion[lesion == 2] = 200
lesion[lesion == 3] = 600
print(lesion.dtype)
count = 0
edgelist = []
for i in lesion:
    count = count +1
    print("count:",count)
    cv2.imwrite('test.png', i)
    img = cv2.imread('test.png')
    canny = cv2.Canny(img,1,150)
    edgelist.append(canny)
edge_array = np.asarray(edgelist)
print("edge_array:",edge_array.shape)
np.save("gt_edge_lge.npy",edge_array)