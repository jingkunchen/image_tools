
import cv2
import numpy as np
import time
import random
import copy
import os
from readimg import readImage 

# def readImage(file_path):
#     img = cv2.imread(file_path)
#     print(img.shape)
#     #img = img[600:700,200:300,0:3]
#     return img


def cropImage(img, h, w, g, jigsaw_n):
    img_tmp = copy.deepcopy(img)
    rowheight = h // jigsaw_n
    colwidth = w // jigsaw_n
    crop_list = []
    for r in range(jigsaw_n):
        for c in range(jigsaw_n):
            c_start = c * colwidth
            c_end = (c + 1) * colwidth
            r_start =  r * rowheight
            r_end = (r + 1) * rowheight
            crop_list.append([c_start,c_end,r_start,r_end])
    rs = random.sample(range(0,jigsaw_n*jigsaw_n),jigsaw_n*jigsaw_n)
    j = 0
    num = 0
    for i in rs:
        img_tmp[crop_list[i][0]:crop_list[i][1],crop_list[i][2]:crop_list[i][3],0:g] = img[crop_list[j][0]:crop_list[j][1],crop_list[j][2]:crop_list[j][3],0:g]
        if i != j:
            num=num + 1
        j += 1
    return img_tmp, num

def cropFromFile(file_dir):
    a = os.walk(file_dir)
    file_list = []
    
    for root, dirs, files in os.walk(file_dir):  
        for file in files: 
            if os.path.splitext(file)[1] == '.tif':  
                file_list.append(os.path.join(root, file))
    or_imageset = []
    crop_imageset = []
    for i in file_list:
        img = readImage(i)
        print(i)
        print(img.shape)
        cv2.imshow('src',img)
        cv2.waitKey(0) 
        or_imageset.append(img)
        crop_img,num  = cropImage(img, 640,640,3,10) 
        # cv2.imshow('src',crop_img)
        # cv2.waitKey(0)
        crop_imageset.append(crop_img)
        return or_imageset, crop_imageset

if __name__ == "__main__":
    file_dir = '/home/chenjingkun/Documents/data/Brodatz/Colored_Brodatz'
    cropFromFile(file_dir)