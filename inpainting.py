from readimg import  readImage,  getFiles, showImage, cropImage
import numpy as np
import random


if __name__ == "__main__":
    file_path = '/home/chenjingkun/Documents/data/uiuc_texture_dataset/train'
    file_list = getFiles(file_path)
    image_list = []
    inpainting_list = []
    for i in file_list:
        img = cropImage(readImage(i),480,480,1)
        image_list.append(img)
        num = random.sample(range(20, 30 ),1)
        rs_1 = random.sample(range(0,480),num[0]*2)
        rs_2 = random.sample(range(5,65),num[0]*2)
        # rs_start = []
        # rs_end = []
        for i in range(0,num[0]):
            if (rs_1[i] +rs_2[i]) > 480:
                rs_start_1 = rs_1[i] - rs_2[i]
                rs_end_1 = rs_1[i]
            else:
                rs_start_1 = rs_1[i]
                rs_end_1 = rs_1[i]+ rs_2[i]
            if (rs_1[i+num[0]] +rs_2[i+num[0]]) > 480:
                rs_start_2 = rs_1[i+num[0]] - rs_2[i+num[0]]
                rs_end_2 = rs_1[i+num[0]]
            else:
                rs_start_2 = rs_1[i+num[0]]
                rs_end_2 = rs_1[i+num[0]]+ rs_2[i+num[0]]
            img[rs_start_1:rs_end_1,rs_start_2:rs_end_2] = 255
            # print(img.shape)
        # showImage(img)
            
        
        

            
            

        
               
        
    
    # data_set =  np.array(image_list)
    # print( data_set.shape)