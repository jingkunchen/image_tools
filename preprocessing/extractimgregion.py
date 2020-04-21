import cv2
import os
import numpy as np
from collections import Counter
import time

segmentation_path = "/Users/chenjingkun/Documents/data/ISBI2016/segmentation/"
img_file = os.listdir(segmentation_path)
size = 32
img_files = []
segmentation_files = []
shake_pixel = 0
for i in img_file:
    segmentation_files.append(segmentation_path+i)
    img_files.append((segmentation_path+i).replace("segmentation","image").replace("_Segmentation.png",".jpg"))
print(len(segmentation_files),len(img_files))

img_health = []
img_lesion = []
for k in range(0,100):
    segmentation = cv2.imread(segmentation_files[k])
    segmentation_shape = segmentation.shape
    segmentation = segmentation[200:segmentation_shape[0]-200,200:segmentation_shape[0]-100,:]
    print(segmentation.shape)
    img = cv2.imread(img_files[k])
    img = img[200:segmentation_shape[0]-200,200:segmentation_shape[0]-100,:]

    # cv2.imshow("test",img)
    # cv2.waitKey(0)

    for i in range(0,int(segmentation.shape[0]/size)):
        for j in range(0,int(segmentation.shape[1]/size)):
            tmp_img = img[shake_pixel+i*size:shake_pixel+(i+1)*size, shake_pixel+j*size:shake_pixel+(j+1)*size,:]
            # print(count, 8+i*32, 8+(i+1)*32, 8+j*32, 8+(j+1)*32)
            tmp_segmentation = segmentation[shake_pixel+i*size:shake_pixel+(i+1)*size, shake_pixel+j*size:shake_pixel+(j+1)*size, :]
            tmp = tmp_segmentation.flatten()
            if(Counter(tmp)[0]>30*30):
                print("health/"+str(k)+'_'+str(shake_pixel+i*size)+'_'+str(shake_pixel+(i+1)*size)+'_'+str(shake_pixel+j*size)+'_'+str(shake_pixel+(j+1)*size)+'_health.jpg')
                img_health.append(tmp_img)
                cv2.imwrite(
                    '{}'.format("health/"+str(k)+'_'+str(shake_pixel+i*size)+'_'+str(shake_pixel+(i+1)*size)+'_'+str(shake_pixel+j*size)+'_'+str(shake_pixel+(j+1)*size)+'_health.jpg'),
                    tmp_img)

            elif(Counter(tmp)[0]<32*8*3):
                # segmentation_list.append(tmp_segmentation)
                img_lesion.append(tmp_img)
                print("lesion/"+str(k)+'_'+str(shake_pixel+i*size)+'_'+str(shake_pixel+(i+1)*size)+'_'+str(shake_pixel+j*size)+'_'+str(shake_pixel+(j+1)*size)+'_lesion.jpg')
                cv2.imwrite(
                    '{}'.format("lesion/"+str(k)+'_'+str(shake_pixel+i*size)+'_'+str(shake_pixel+(i+1)*size)+'_'+str(shake_pixel+j*size)+'_'+str(shake_pixel+(j+1)*size)+'_lesion.jpg'),
                    tmp_img)
                
img_health_array = np.asarray(img_health)
img_lesion_array = np.asarray(img_lesion)
img_health_array = img_health_array/255.
img_lesion_array = img_lesion_array/255.
print("img_health_array:",img_health_array.shape)
print("img_lesion_array:",img_lesion_array.shape)
np.save("skin_health_array_32_32_train_25_100.npy", img_health_array[img_lesion_array.shape[0]:,:,:,:])
np.save("skin_health_array_32_32_test_25_100.npy", img_health_array[:img_lesion_array.shape[0],:,:,:])
np.save("skin_lesion_array_32_32_test_25_100.npy", img_lesion_array)
        

# for i in img_files:
    # print("img_files:",i)
    # tmp = cv2.imread(i)
    # img_list.append(tmp)
# for i in segmentation_files:
    # print("segmentation_files:",i)
    # tmp = cv2.imread(i)
    # print(tmp.shape)
    # segmentation_list.append(tmp)

# img_array = np.asarray(img_list)
# segmentation_array = np.asarray(segmentation_list)
# print("img_array:",img_array.shape)
# print("segmentation_array:",segmentation_array.shape)
    # cv2.imshow("tmp",tmp)
    # cv2.waitKey(0)


# count = 0
# new_health_array = []
# for img in gt_array:
#     if np.all(img == 0):
#         pass
#     else:
#         for i in range(0,7):
#             for j in range(0,7):
#                 tmp_img = img[shake_pixel+i*32:shake_pixel+(i+1)*32,shake_pixel+j*32:shake_pixel+(j+1)*32]
#                 if np.all(tmp_img == 0):
#                     # print(count, 8+i*32, 8+(i+1)*32, 8+j*32, 8+(j+1)*32)
#                     tmp_data = data_array[count, shake_pixel+i*32:shake_pixel+(i+1)*32, shake_pixel+j*32:shake_pixel+(j+1)*32]
#                     tmp = tmp_data.flatten()
#                     tmp_array_1 = data_array_1[count, shake_pixel+i*32:shake_pixel+(i+1)*32, shake_pixel+j*32:shake_pixel+(j+1)*32]
#                     # print("health_Counter(tmp)[0]:",Counter(tmp)[0])
#                     if(Counter(tmp)[0]>10):
#                         pass
#                     else:
#                         new_health_array.append(tmp_array_1)
#                         io.imsave(
#                             fname='{}{}'.format(health_path, str(count)+'_'+str(shake_pixel+i*32)+'_'+str(shake_pixel+(i+1)*32)+'_'+str(shake_pixel+j*32)+'_'+str(shake_pixel+(j+1)*32)+'.jpg'),
#                             arr=tmp_array_1)
#     count = count + 1