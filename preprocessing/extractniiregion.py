import os
import SimpleITK as sitk
import skimage.io as io
import scipy.ndimage
import scipy.misc
import time
import cv2
import numpy as np
from collections import  Counter
import pandas as pd


health_path = '/Users/chenjingkun/Documents/data/BraTS19/health/'
lesion_path = '/Users/chenjingkun/Documents/data/BraTS19/lesion/'
nii_path = '/Users/chenjingkun/Documents/data/BraTS19/HGG/'
csv_path = '/Users/chenjingkun/Documents/data/BraTS19/name_mapping.csv'
shake_pixel = 8
cut_size = 32
def show_img(data):
    for i in range(data.shape[0]):
        print(np.max(data[i, :, :]))
        print(np.min(data[i, :, :]))
        io.imshow(data[i, :, :], cmap='gray')
        io.show()
def show_img_single(data):
        io.imshow(data[:,:], cmap = 'gray')
        io.show()
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def readCsv(file_path):
    csv_data = pd.read_csv(file_path)  # 读取训练数据
    print(csv_data.shape)  # (189, 9)
    # csv_batch_data = csv_data.tail(N)  # 取后5条数据
    csv_batch_data = csv_data # 取后5条数据
    train_batch_data = csv_batch_data['BraTS_2019_subject_ID']  # 取这20条数据的3到5列值(索引从0开始)
    return train_batch_data

def nii2NpySlice(data_path, gt_path):
    thresh = 1
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_path)))
    data_array_1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path)))

    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0
    for iii in range(np.shape(data_array)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(mask[iii, :, :])  #fill the holes inside br

    data_array_1 = data_array_1 - np.mean(data_array[mask == 1])
    data_array_1 /= np.std(data_array_1[mask == 1])
    data_array_1 = normalize(data_array_1)
    
    count = 0
    new_health_array = []
    for img in gt_array:
        if np.all(img == 0):
            pass
        else:
            for i in range(0,7):
                for j in range(0,7):
                    tmp_img = img[shake_pixel+i*cut_size:shake_pixel+(i+1)*cut_size,shake_pixel+j*cut_size:shake_pixel+(j+1)*cut_size]

                    if np.all(tmp_img == 0):
                        # print(count, 8+i*32, 8+(i+1)*32, 8+j*32, 8+(j+1)*32)
                        print("data_array:",data_array.shape)
                        tmp_data = data_array[count, shake_pixel+i*cut_size:shake_pixel+(i+1)*cut_size, shake_pixel+j*cut_size:shake_pixel+(j+1)*cut_size]
                        print("tmp_data:",tmp_data.shape)
                        tmp = tmp_data.flatten()
                        tmp_array_1 = data_array_1[count, shake_pixel+i*cut_size:shake_pixel+(i+1)*cut_size, shake_pixel+j*cut_size:shake_pixel+(j+1)*cut_size]
                        print("tmp_array_1:",tmp_array_1.shape)

                        if(Counter(tmp)[0]>10):
                            pass
                        else:
                            
                            new_health_array.append(tmp_array_1)
                            io.imsave(
                                fname='{}{}'.format(health_path, str(count)+'_'+str(shake_pixel+i*cut_size)+'_'+str(shake_pixel+(i+1)*cut_size)+'_'+str(shake_pixel+j*cut_size)+'_'+str(shake_pixel+(j+1)*cut_size)+'.jpg'),
                                arr=tmp_array_1)
        count = count + 1

    count = 0
    new_lesion_array = []
    for gt in gt_array:
        if np.all(gt == 0):
            pass
        else:
            for i in range(0,7):
                for j in range(0,7):
                    tmp_gt = gt[shake_pixel+i*cut_size:shake_pixel+(i+1)*cut_size,shake_pixel+j*cut_size:shake_pixel+(j+1)*cut_size]

                    tmp_array_1 = data_array_1[count, shake_pixel+i*cut_size:shake_pixel+(i+1)*cut_size, shake_pixel+j*cut_size:shake_pixel+(j+1)*cut_size]
 
                    if np.all(tmp_gt == 0):
                        pass
                    else:
                        tmp_check = tmp_array_1.flatten()
                        if(Counter(tmp_check)[0]>10):
                            pass
                        else:
                            tmp = tmp_gt.flatten()
                            #256-1024, 512-1024, 768-1024, 1014-1024
                            if(Counter(tmp)[0]>256):
                                pass
                            else:
                                new_lesion_array.append(tmp_array_1)
                                io.imsave(
                                    fname='{}{}'.format(lesion_path, str(count)+'_'+str(shake_pixel+i*32)+'_'+str(shake_pixel+i*32+32)+'_'+str(shake_pixel+j*32)+'_'+str(shake_pixel+j*32+32)+'.jpg'),
                                    arr=tmp_array_1
                                )
                            
        count = count + 1
    new_health_data = np.asarray(new_health_array)
    # np.save('health.npy', new_health_data)
    new_lesion_data = np.asarray(new_lesion_array)
    return new_health_data, new_lesion_data

def main():
    file_name_list = readCsv(csv_path)
    data_path_list = []
    gt_path_list = []
    for i in file_name_list:
        data_path_list.append(nii_path+i+'/'+i+'_flair.nii.gz')
        gt_path_list.append(nii_path+i+'/'+i+'_seg.nii.gz')
    print(data_path_list)
    print(gt_path_list)
    health_array, lesion_array = nii2NpySlice(data_path_list[0], gt_path_list[0])

    for i in range(len(data_path_list)-1):
        print(i)
        tmp_health_array, tmp_lesion_array = nii2NpySlice(data_path_list[i+1], gt_path_list[i+1])
        print(health_array.shape,tmp_health_array.shape)
        print(lesion_array.shape,tmp_lesion_array.shape)
        try:
            health_array = np.concatenate((health_array, tmp_health_array), axis=0)
            lesion_array = np.concatenate((lesion_array, tmp_lesion_array), axis=0)
            print("health_array:",health_array.shape)
            print("lesion_array:",lesion_array.shape)
        except:
            print("------------error----------------")
    print("health_array:",health_array.shape)
    print("lesion_array:",lesion_array.shape)
    np.save('brain_health_array_32_32_train.npy', health_array[lesion_array.shape[0]:,:,:,np.newaxis])
    np.save('brain_lesion_array_32_32_test.npy', lesion_array[:,:,:,np.newaxis])
    np.save('brain_health_array_32_32_test.npy', health_array[:lesion_array.shape[0],:,:,np.newaxis])
    
        

if __name__ == '__main__':
   
    main()
    
    # np.save('lesion.npy', new_lesion_data)
# test = np.load('lesion.npy')
# for i in test:
#     cv2.imshow("Image",i)
#     cv2.waitKey (0)