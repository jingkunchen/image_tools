import os
import cv2
import numpy as np

def add_mask2image_binary(images_path, masked_path, masked_filename, points_x, points_y):
# Add binary masks to images
   
    img = cv2.imread(images_path)
    color = [255, 255, 255]
    points = []
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    for i in range(len(points_x)):
        points.append((points_x[i],points_y[i]))

    img_out = cv2.fillPoly(img, [np.array(points)], color)
    # print(img.shape, img_out.shape)
    cv2.imwrite(masked_path+'out_'+ masked_filename, img_out) 
   
    bg = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    img_in = cv2.fillPoly(bg, [np.array(points)], color)
    cv2.imwrite(masked_path+'mask_'+ masked_filename, img_in) 

    img = cv2.imread(images_path)
    masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=img_in) 
    cv2.imwrite(masked_path+'masked_'+ masked_filename, masked) 

images_path = '/Users/chenjingkun/Documents/data/COVID/normal/weixian/original/'
masked_path = '/Users/chenjingkun/Documents/data/COVID/normal/weixian/masked/'


import json
with open("/Users/chenjingkun/Documents/data/COVID/normal/weixian/original.json", 'r') as f:
    tmp = f.read()
    temp = json.loads(tmp)
    count = 0

    for key, value in temp.items(): 
        # try:
            count = count + 1
            
            images_file = images_path + value['filename']
            print(images_file, count)
            masked_path = masked_path
            points_x = value['regions'][0]['shape_attributes']['all_points_x']
            points_y = value['regions'][0]['shape_attributes']['all_points_y']
            add_mask2image_binary(images_file, masked_path, value['filename'], points_x, points_y)
        # except:
           
    
