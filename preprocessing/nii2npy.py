from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
import cv2
import time
from decimal import Decimal
import skimage.io as io
from skimage.morphology import square
from skimage.morphology import dilation

thresh = 1
rows = 224
cols = 224
xmin = 1 
xmax = 1
ymin = 1
ymax = 1
xlenmin = 1
ylenmin = 1

img_count = 0
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        io.show()
def show_img_single(data):
        io.imshow(data[:,:], cmap = 'gray')
        io.show()


# label transform, 500-->1, 200-->2, 600-->3

data_1ch = []
gt_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/stacom/c0'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/stacom/c0gt'
lge_list = []
for pp in range(1, 46):

    data_name = img_dir + '/patient' + str(pp) + '_C0.nii.gz'
    gt_name = gt_dir_1 + '/patient' + str(pp) + '_C0_manual.nii.gz'
    img = sitk.ReadImage(os.path.join(gt_name))
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    
    img_count += gt_array.shape[0]
    # show_img(gt_array)
    print(np.shape(data_array))
    print(np.shape(gt_array))

    x = []
    y = []
    print("idx:", pp)
    for image in gt_array:
        # tmp = dilation(image, square(3))
        # show_img_single(tmp)
        for i in range(np.shape(gt_array)[1]):
            for j in range(np.shape(gt_array)[2]):
                if image[i][j] != 0:
                    if i <30 or j<30:
                        print("label_error:", pp,i,j,image[i][j])
                    else:
                        x.append(i)
                        y.append(j)
    print(min(x),max(x),max(x)-min(x),round(min(x)/np.shape(gt_array)[1],2), round(max(x)/np.shape(gt_array)[1],2))
    print(min(y),max(y),max(y)-min(y),round(min(y)/np.shape(gt_array)[1],2), round(max(y)/np.shape(gt_array)[1],2))

    # if gt_array.shape[1] == 480 or gt_array.shape[1] == 512:
    #     data_array = data_array[:,136:360,136:360]
    #     gt_array = gt_array[:,136:360,136:360]
    # elif int(gt_array.shape[1]) == 400:
    #     data_array = data_array[:,88:312,88:312]
    # elif int(gt_array.shape[1]) == 432:
    #     data_array = data_array[:,104:328,104:328]
    # elif gt_array.shape[1] == 224:
    #     pass
    # else:
    #     print("error:",gt_array.shape, int(gt_array.shape[1]) == 400)
    
    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0
    for iii in range(np.shape(data_array)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br
    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]

    data_array_ = data_array[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    gt_array_ = gt_array[:,
                         int((rows_o - rows) /
                             2):int((rows_o - rows) / 2) + rows,
                         int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
                         cols]
    mask = mask[:,
                int((rows_o - rows) / 2):int((rows_o - rows) / 2) + rows,
                int((cols_o - cols) / 2):int((cols_o - cols) / 2) + cols]

    data_1ch.extend(np.float32(data_array_))
    gt_1ch.extend(np.float32(gt_array_))


data_1ch = np.asarray(data_1ch)
gt_1ch = np.asarray(gt_1ch)
print("data_1ch:",data_1ch.shape)
data_1ch = data_1ch[:,:,:,np.newaxis]
gt_1ch = gt_1ch[:,:,:,np.newaxis]
print("data_1ch:",data_1ch.shape)
gt_1ch[gt_1ch == 500] = 1
gt_1ch[gt_1ch == 200] = 1
gt_1ch[gt_1ch == 600] = 1
np.save('C0_data_1.npy', data_1ch)
np.save('C0_gt_1.npy', gt_1ch)
print("C0_gt:",gt_1ch.shape)
print(img_count)