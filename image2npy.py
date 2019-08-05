import os
import numpy as np
import cv2

img_dir = "/Users/chenjingkun/Documents/data/skin_32_32/lesiontest"

if not os.path.exists(img_dir):
    os.makedirs(img_dir)



filename = os.listdir(img_dir)
file_list = []
for i in filename:
    file_list.append(img_dir +'/'+ i)
img_array = cv2.imread(file_list[0])[np.newaxis,:,:,:]
print(img_array.shape)
for i in file_list[1:]:
    print(i)
    img = cv2.imread(i)[np.newaxis,:,:,:]
    # print(img.shape)
    img_array = np.concatenate([img_array, img], axis = 0)
    print(img_array.shape)

np.save("skin_32_32_lesion_test.npy",img_array)
    
