import os
import numpy as np
import cv2


import random
import copy
import io
from scipy import stats

width = 32
hight = 32
width_start = 10
hight_start = 10
channel = 1

def show_img(data):
    for i in range(data.shape[0]):
        cv2.imshow('src', data[i, :, :, :])
        cv2.waitKey(0)

def mergeNpy(img_set1, img_set2):
    img_set = np.concatenate([img_set1, img_set2], axis = 0)
    return img_set

def zscore_calculate(img_dir):
    i = 0
    for root1, dirs1, files1 in os.walk(img_dir):  
        root = root1
        file = files1
    file_list = []
    for i in file:
        file_list.append(root+'/'+i)
    print(file_list[0])
    # file_list = getFiles(img_dir)
    tmp = cv2.imread(file_list[0])
    tmp = cv2.resize(tmp[:,:,0], (512,512), interpolation=cv2.INTER_AREA) 
    print(tmp.shape)
    all_image = tmp[np.newaxis,:,:]
    for i in file_list[1:]:
        print(i)
        tmp = cv2.imread(i)
      
        tmp = cv2.resize(tmp[:,:,0], (512,512), interpolation=cv2.INTER_AREA) 
        # cv2.imshow("lesion.png",tmp)
        # cv2.waitKey(0)
        # print(i)
        all_image = mergeNpy(all_image, tmp[np.newaxis,:,:])
        print(all_image.shape)
    X_scaled = stats.zscore(all_image)
    return X_scaled

    # img = cropImage(readImage(file_list[0]), start_width, start_hight, width, hight, channel)
    # img_array = img[np.newaxis,:,:,:]
    # for i in file_list[1:]:
          # img = cropImage(readImage(i), start_width, start_hight, width, hight, channel)
    #     img = img[np.newaxis,:,:,:]        
    #     img_array = mergeNpy(img_array,img)
    # return img_array

def inPaintingNpy(img_set):
    img_set_tmp = copy.deepcopy(img_set)
    count = img_set_tmp.shape[0]
    print("len:",len(img_set_tmp))
    for j in range(count):
        num = random.sample(range(5, 10),1)
        print("num:",num)
        rs_1 = random.sample(range(0,hight-3),num[0]*2)
        print("rs_1:",rs_1)
        rs_2 = [random.randint(3, 6) for __ in range(num[0]*2)]
        # rs_2 = random.sample(range(20,40),num[0]*2)
        # print("rs_2:",rs_2)
        # file_num = random.sample(range(1,count),num[0])
        # print("file_num:",file_num)
        for i in range(0,num[0]):
            if (rs_1[i] +rs_2[i]) > width:
                rs_start_1 = rs_1[i] - rs_2[i]
                rs_end_1 = rs_1[i]
            else:
                rs_start_1 = rs_1[i]
                rs_end_1 = rs_1[i]+ rs_2[i]
            if (rs_1[i+num[0]] +rs_2[i+num[0]]) > width:
                rs_start_2 = rs_1[i+num[0]] - rs_2[i+num[0]]
                rs_end_2 = rs_1[i+num[0]]
            else:
                rs_start_2 = rs_1[i+num[0]]
                rs_end_2 = rs_1[i+num[0]]+ rs_2[i+num[0]]
            # img_set_tmp[j,rs_start_1:rs_end_1,rs_start_2:rs_end_2,:] = img_set[file_num[i],rs_start_1:rs_end_1,rs_start_2:rs_end_2,:]
            img_set_tmp[j,rs_start_1:rs_end_1,rs_start_2:rs_end_2,:] = 0
    return img_set_tmp

def outPaintingNpy(img_set):
    img_set_tmp = copy.deepcopy(img_set)
    count = len(img_set_tmp)
    for j in range(count):
        num = random.sample(range(10, 20),10)
        
        rs_1_w = random.sample(range(0,width),num[0])
        rs_1_w_2 = random.sample(range(0,60),num[0])
        rs_1_h = random.sample(range(0,194),num[0])

        rs_2_w = random.sample(range(0,width),num[1])
        rs_2_w_2 = random.sample(range(164,width),num[1])
        rs_2_h = random.sample(range(0,194),num[1])

        rs_3_w = random.sample(range(0,width),num[2])
        rs_3_w_2 = random.sample(range(0,60),num[2])
        rs_3_h = random.sample(range(0,194),num[2])

        rs_4_w = random.sample(range(0,width),num[3])
        rs_4_w_2 = random.sample(range(164,width),num[3])
        rs_4_h = random.sample(range(0,194),num[3])

        for i in range(0,num[0]):
            file_num = random.sample(range(1,count),num[0])
            tmp = random.sample(range(10, 30),1)[0]
            img_set_tmp[j,0:rs_1_w_2[i],rs_1_h[i]:rs_1_h[i]+tmp,:] = img_set[file_num[i],0:rs_1_w_2[i],rs_1_h[i]:rs_1_h[i]+tmp,:]
            # img[0:rs_1_w_2[i],rs_1_h[i]:rs_1_h[i]+random.sample(range(10, 30),1)[0]] = 1
        for i in range(0,num[1]):
            file_num = random.sample(range(1,count),num[1])
            tmp = random.sample(range(10, 30),1)[0]
            img_set_tmp[j,rs_2_w_2[i]:width,rs_2_h[i]:rs_2_h[i]+tmp,:] = img_set[file_num[i],rs_2_w_2[i]:width,rs_2_h[i]:rs_2_h[i]+tmp,:]
        for i in range(0,num[2]):
            file_num = random.sample(range(1,count),num[2])
            tmp = random.sample(range(10, 30),1)[0]
            img_set_tmp[j,rs_3_h[i]:rs_3_h[i]+tmp,0:rs_3_w_2[i],:] = img_set[file_num[i],rs_3_h[i]:rs_3_h[i]+tmp,0:rs_3_w_2[i],:]
        for i in range(0,num[3]):
            file_num = random.sample(range(1,count),num[3])
            tmp = random.sample(range(10, 30),1)[0]
            img_set_tmp[j,rs_4_h[i]:rs_4_h[i]+tmp,rs_4_w_2[i]:width,:] = img_set[file_num[i],rs_4_h[i]:rs_4_h[i]+tmp,rs_4_w_2[i]:width,:]
        # showImage(img_set_tmp[j,:,:,:])
    return img_set_tmp

if __name__ == "__main__":
    translate = 1
    painting = 0
    merge = 0
    img_list = []
    if translate:
        img_dir = '/Users/chenjingkun/Documents/data/COVID/covid/original'
        img_array = zscore_calculate(img_dir)
        
        # for i in img_array:
        #     # tmp = cropImage(i, width_start, width_start + width, hight_start, hight_start + hight, channel)
        #     cv2.imshow("1pyt",i)
        #     img_list.append(i)

        crop_array = np.asarray(img_list)
        
        np.save("covid_original.npy",img_array)

    if painting:
        npy_file = "skin_health_array_32_32_train_25_100.npy"
        img_set = np.load(npy_file)[:,:,:,:]
        # img_set = img_set[:,:,:,np.newaxis]
        new_img_set = inPaintingNpy(img_set)
        print(new_img_set.shape)
        # print(new_img_set.shape)
        # show_img(new_img_set)
        np.save("skin_health_array_32_32_train_25_100_inpainting.npy",new_img_set)

    if merge:
        npy_path1 = "/Users/chenjingkun/Documents/data/texture/uiuc_texture_train.npy"
        npy_path2 = "/Users/chenjingkun/Documents/data/texture/uiuc_texture_train.npy"
        npy_file1 = np.load(npy_path1)
        npy_file2 = np.load(npy_path2)
        print("npy_file1:",npy_file1.shape)
        print("npy_file2:",npy_file2.shape)
        npy_file = mergeNpy(npy_file1,npy_file2)
        print(npy_file.shape)
        np.save("uiuc_texture_train_double.npy", npy_file)
