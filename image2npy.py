import os
import numpy as np
import cv2
from readimg import  readImage,  getFiles, showImage, cropImage
from inpainting import inPainting, outPainting


img_dir = '/home/chenjingkun/Documents/data/uiuc_texture_dataset/test'

file_list = getFiles(img_dir)
# print(file_list[0])
img =  cropImage(readImage(file_list[0]),224,224,1)
img_array = img[np.newaxis,:,:,:]   
# print(img_array.shape)
for i in file_list[1:]:
    # print(i)
    img = cropImage(readImage(i), 224, 224,1)
    # img = inPainting(img)
    img = outPainting(img)
    showImage(img)
    # print("image2npy:",img)
    img = img[np.newaxis,:,:,:]
    
    # print(img.shape)
    img_array = np.concatenate([img_array, img], axis = 0)
    print(img_array.shape)
np.save("texture_test_inpainting.npy",img_array)
    
