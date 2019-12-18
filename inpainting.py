from readimg import  readImage,  getFiles, showImage, cropImage
import numpy as np
import random


def inPainting(img,length):
    num = random.sample(range(5, 15),1)
    rs_1 = random.sample(range(10,length-10),num[0]*2)
    rs_2 = random.sample(range(5,35),num[0]*2)
    # rs_start = []
    # rs_end = []
    for i in range(0,num[0]):
        if (rs_1[i] +rs_2[i]) > length:
            rs_start_1 = rs_1[i] - rs_2[i]
            rs_end_1 = rs_1[i]
        else:
            rs_start_1 = rs_1[i]
            rs_end_1 = rs_1[i]+ rs_2[i]
        if (rs_1[i+num[0]] +rs_2[i+num[0]]) > length:
            rs_start_2 = rs_1[i+num[0]] - rs_2[i+num[0]]
            rs_end_2 = rs_1[i+num[0]]
        else:
            rs_start_2 = rs_1[i+num[0]]
            rs_end_2 = rs_1[i+num[0]]+ rs_2[i+num[0]]
        img[rs_start_1:rs_end_1,rs_start_2:rs_end_2] = 1
    return img

def outPainting(img):
    num = random.sample(range(3, 8),4)
    rs_1_w = random.sample(range(0,224),num[0])
    rs_1_w_2 = random.sample(range(0,60),num[0])
    rs_1_h = random.sample(range(0,194),num[0])

    rs_2_w = random.sample(range(0,224),num[1])
    rs_2_w_2 = random.sample(range(164,224),num[1])
    rs_2_h = random.sample(range(0,194),num[1])

    rs_3_w = random.sample(range(0,224),num[2])
    rs_3_w_2 = random.sample(range(0,60),num[2])
    rs_3_h = random.sample(range(0,194),num[2])

    rs_4_w = random.sample(range(0,224),num[3])
    rs_4_w_2 = random.sample(range(164,224),num[3])
    rs_4_h = random.sample(range(0,194),num[3])


    for i in range(0,num[0]):
        img[0:rs_1_w_2[i],rs_1_h[i]:rs_1_h[i]+random.sample(range(10, 30 ),1)[0]] = 1
    for i in range(0,num[1]):
        img[rs_2_w_2[i]:224,rs_2_h[i]:rs_2_h[i]+random.sample(range(10, 30 ),1)[0]] = 1
    for i in range(0,num[2]):
        img[rs_3_h[i]:rs_3_h[i]+random.sample(range(10, 30),1)[0],0:rs_3_w_2[i]] = 1
    for i in range(0,num[3]):
        img[rs_4_h[i]:rs_4_h[i]+random.sample(range(10, 30 ),1)[0],rs_4_w_2[i]:224] = 1
    
    # for i in range(0,num[1]):
    #     img[0:rs_2_w[i],rs_2_h[i]:0] = 255
    # for i in range(0,num[2]):
    #     img[rs_3_w[i]:0,rs_3_h[i]:0] = 255
    # for i in range(0,num[3]):
    #     img[rs_4_w[i]:0,0:rs_4_h[i]] = 255
    return img

if __name__ == "__main__":
    file_path = '/home/chenjingkun/Documents/data/uiuc_texture_dataset/train'
    file_list = getFiles(file_path)
    image_list = []
    inpainting_list = []
    for i in file_list:
        img = cropImage(readImage(i),224,224,1)
        img = outPainting(img)
        showImage(img)
        # image_list.append(img)
        # num = random.sample(range(20, 30 ),1)
        # rs_1 = random.sample(range(0,480),num[0]*2)
        # rs_2 = random.sample(range(5,65),num[0]*2)
        # # rs_start = []
        # # rs_end = []
        # for i in range(0,num[0]):
        #     if (rs_1[i] +rs_2[i]) > 480:
        #         rs_start_1 = rs_1[i] - rs_2[i]
        #         rs_end_1 = rs_1[i]
        #     else:
        #         rs_start_1 = rs_1[i]
        #         rs_end_1 = rs_1[i]+ rs_2[i]
        #     if (rs_1[i+num[0]] +rs_2[i+num[0]]) > 480:
        #         rs_start_2 = rs_1[i+num[0]] - rs_2[i+num[0]]
        #         rs_end_2 = rs_1[i+num[0]]
        #     else:
        #         rs_start_2 = rs_1[i+num[0]]
        #         rs_end_2 = rs_1[i+num[0]]+ rs_2[i+num[0]]
        #     img[rs_start_1:rs_end_1,rs_start_2:rs_end_2] = 255
            # print(img.shape)
        # showImage(img)
            
        
        

            
            

        
               
        
    
    # data_set =  np.array(image_list)
    # print( data_set.shape)