import os
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import skimage.io as io
import scipy.misc

def show_img(data):
        io.imshow(data[:,:], cmap = 'gray')
        io.show()

rows = 224
cols = 224
start_number = 1
end_number = 45

thresh = 1
data_dir_c0 = '/Users/chenjingkun/Documents/data/stacom/c0/'
gt_dir_c0 = '/Users/chenjingkun/Documents/data/stacom/c0gt/'
data_dir_lge = '/Users/chenjingkun/Documents/data/stacom/lge/'
gt_dir_lge = '/Users/chenjingkun/Documents/data/stacom/lgegt/'

LGE_slice_number = []
C0_slice_number = []

lge_slice = []
lge_gt = []
c0_slice = []
c0_gt = []


for pp in range(start_number, end_number+1):
    print("number:",pp)
    data_name_lge = data_dir_lge + 'patient' + str(pp) + '_LGE.nii.gz'
    data_array_lge_tmp = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_name_lge)))
    data_array_lge_tmp = np.nan_to_num(data_array_lge_tmp, copy=True)
    gt_name_lge = gt_dir_lge + 'patient' + str(pp) + '_LGE_manual.nii.gz'
    gt_array_lge_tmp = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name_lge)))
    gt_array_lge_tmp = np.nan_to_num(gt_array_lge_tmp, copy=True)

    data_name_c0 = data_dir_c0 + 'patient' + str(pp) + '_C0.nii.gz'
    data_array_c0_tmp = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_name_c0)))
    data_array_c0_tmp = np.nan_to_num(data_array_c0_tmp, copy=True)
    gt_name_c0 = gt_dir_c0 + 'patient' + str(pp) + '_C0_manual.nii.gz'
    gt_array_c0_tmp = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name_c0)))
    gt_array_c0_tmp = np.nan_to_num(gt_array_c0_tmp, copy=True)


    mask = np.zeros(np.shape(data_array_lge_tmp), dtype='float32')
    mask[data_array_lge_tmp >= thresh] = 1
    mask[data_array_lge_tmp < thresh] = 0
    for iii in range(np.shape(data_array_lge_tmp)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br
    data_array_lge_tmp = data_array_lge_tmp - np.mean(data_array_lge_tmp[mask == 1])
    data_array_lge_tmp /= np.std(data_array_lge_tmp[mask == 1])
    rows_o = np.shape(data_array_lge_tmp)[1]
    cols_o = np.shape(data_array_lge_tmp)[2]
    data_array_lge_tmp = data_array_lge_tmp[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    gt_array_lge_tmp = gt_array_lge_tmp[:,
                         int((rows_o - rows) /
                             2):int((rows_o - rows) / 2) + rows,
                         int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
                         cols]
    lge_slice.append(np.float32(data_array_lge_tmp))
    lge_gt.append(np.float32(gt_array_lge_tmp))
    LGE_slice_number.append(data_array_lge_tmp.shape[0])
    
    mask = np.zeros(np.shape(data_array_c0_tmp), dtype='float32')
    mask[data_array_c0_tmp >= thresh] = 1
    mask[data_array_c0_tmp < thresh] = 0
    for iii in range(np.shape(data_array_c0_tmp)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br
    data_array_c0_tmp = data_array_c0_tmp - np.mean(data_array_c0_tmp[mask == 1])
    data_array_c0_tmp /= np.std(data_array_c0_tmp[mask == 1])
    rows_o = np.shape(data_array_c0_tmp)[1]
    cols_o = np.shape(data_array_c0_tmp)[2]
    data_array_c0_tmp = data_array_c0_tmp[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    gt_array_c0_tmp = gt_array_c0_tmp[:,
                         int((rows_o - rows) /
                             2):int((rows_o - rows) / 2) + rows,
                         int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
                         cols]
    c0_slice.append(np.float32(data_array_c0_tmp))
    c0_gt.append(np.float32(gt_array_c0_tmp))
    C0_slice_number.append(gt_array_c0_tmp.shape[0])

count_list = []
for i in range(start_number-1,end_number):
    tmp_list = []
    tmp = float(LGE_slice_number[i])/float(C0_slice_number[i])
    for j in range(0,C0_slice_number[i]):
        tmp_list.append(round(tmp*j))
    print("tmp_list:", i, tmp_list)
    count_list.append(tmp_list)

data_list_lge = []
gt_list_lge = []
for i in range(start_number-1, end_number):
    for j in count_list[i]:
        while(j > len(lge_slice[i]) or j == len(lge_slice[i])):
            j = j-1
        print(i, j, len(lge_slice[i]))

        data_list_lge.append(lge_slice[i][j])
        gt_list_lge.append(lge_gt[i][j])
data_list_c0 = []
gt_list_c0 = []        
for i in range(start_number-1, end_number):
    for j in range(0,len(c0_slice[i])):
        data_list_c0.append(c0_slice[i][j])
        gt_list_c0.append(c0_gt[i][j])
data_array_lge = np.array(data_list_lge)  
gt_array_lge = np.array(gt_list_lge) 
data_array_c0 = np.array(data_list_c0)  
gt_array_c0 = np.array(gt_list_c0) 
gt_array_lge[gt_array_lge == 500] = 1
gt_array_lge[gt_array_lge == 200] = 2
gt_array_lge[gt_array_lge == 600] = 3

gt_array_c0[gt_array_c0 == 500] = 1
gt_array_c0[gt_array_c0 == 200] = 2
gt_array_c0[gt_array_c0 == 600] = 3
print("data_array_lge:",data_array_lge.shape)
print("gt_array_lge:",gt_array_lge.shape)
print("data_array_c0:",data_array_c0.shape)
print("gt_array_c0:",gt_array_c0.shape)
np.save('data_array_lge.npy', data_array_lge)
np.save('gt_array_lge.npy', gt_array_lge)
np.save('data_array_c0.npy', data_array_c0)
np.save('gt_array_c0.npy', gt_array_c0)

# x = x.astype("float32")
# print(x.shape)
# x = x
sitk.WriteImage(sitk.GetImageFromArray(data_array_lge), "data_array_lge.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(gt_array_lge), "gt_array_lge.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(data_array_c0), "data_array_c0.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(gt_array_c0), "gt_array_c0.nii.gz")




#     mask = np.zeros(np.shape(data_array), dtype='float32')
#     mask[data_array >= thresh] = 1
#     mask[data_array < thresh] = 0

#     data_array = data_array - np.mean(data_array[mask == 1])
#     data_array /= np.std(data_array[mask == 1])
#     rows_o = np.shape(data_array)[1]
#     cols_o = np.shape(data_array)[2]

#     data_array_ = data_array[:,
#                              int((rows_o - rows) /
#                                  2):int((rows_o - rows) / 2) + rows,
#                              int((cols_o - cols) /
#                                  2):int((cols_o - cols) / 2) + cols]
#     new_count_x_list.append(int((rows_o - rows) /2))
#     new_count_y_list.append(int((cols_o - cols) /2))
    
#     LGE_data_1ch.extend(np.float32(data_array_))

# LGE_data_1ch = np.asarray(LGE_data_1ch)
# # sitk.WriteImage(sitk.GetImageFromArray(LGE_data_1ch),"test_image.nii.gz")
# # np.save('transfer_T2_data.npy', LGE_data_1ch)

# T2_count = 0
# for pp in range(start_number, end_number):
#     new_count_x = new_count_x_list[pp-start_number]
#     new_count_y = new_count_y_list[pp-start_number]
#     new_shape = (LGE_shape[pp-start_number][1],LGE_shape[pp-start_number][2])
#     gt_name = gt_dir_1 + 'patient' + str(pp) + '_C0_manual.nii.gz'
#     gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
#     gt_array = np.nan_to_num(gt_array, copy=True)
#     print(gt_array.shape)
   
#     x = []
#     y = []
#     count = 0
#     print("idx:", pp)
#     new_gt_list = []
#     for image in gt_array:
        
#         image = np.asarray(image)
#         # show_img(image)
#         image1 = image.copy()
#         image2 = image.copy()
#         image[image == 500] = 1
#         image[image == 200] = 0
#         image[image == 600] = 0
#         image1[image1 == 500] = 0
#         image1[image1 == 200] = 1
#         image1[image1 == 600] = 0
#         image2[image2 == 500] = 0
#         image2[image2 == 200] = 0
#         image2[image2 == 600] = 1
        
#         image = resize(image,new_shape, preserve_range =True)
#         image1 = resize(image1,new_shape, preserve_range =True)
#         image2 = resize(image2,new_shape, preserve_range =True)

#         image = np.around(image)
#         image1 = np.around(image1)
#         image2 = np.around(image2)
#         image = image.astype(np.int32)
#         image1 = image1.astype(np.int32)
#         image2 = image2.astype(np.int32)
        
#         image[image == 1] = 1
#         image1[image1 == 1] = 2
#         image2[image2 == 1] = 3
#         image = image +image1 +image2
#         [x_test, y_test] = image.shape
#         for i in range(x_test):
#             for j in range(y_test):
#                 if(image[i, j] >3) :
#                     print("--------error----------:", pp, count)
#         image[image == 1] = 500
#         image[image == 2] = 200
#         image[image == 3] = 600
        
#         for i in range(np.shape(gt_array)[1]):
#             for j in range(np.shape(gt_array)[2]):
#                 if image[i][j] != 0:
#                     if j < 40 or i < 40:
#                         gt_array[count, 0:75, 0:50] = 0
#                         image[0:200, 0:50] = 0
#                     else:
#                         x.append(i)
#                         y.append(j)
#         new_gt_list.append(image)
#         print("new_gt_list:",len(new_gt_list))
                    
#         count += 1
#     gt_array=np.array(new_gt_list)
#     print("new_array:",gt_array.shape)
    
#     print(min(x), max(x),
#           max(x) - min(x), round(min(x) / np.shape(gt_array)[1], 2),
#           round(max(x) / np.shape(gt_array)[1], 2))
#     print(min(y), max(y),
#           max(y) - min(y), round(min(y) / np.shape(gt_array)[1], 2),
#           round(max(y) / np.shape(gt_array)[1], 2))
#     if(round(min(x)/np.shape(gt_array)[1],2) < 0.2 or round(min(y)/np.shape(gt_array)[1],2)<0.2):
#         print("errorerrorerrorerrorerrorerror")
#         show_img(gt_array)
#     #C0
#     gt_array_ = gt_array[:, new_count_x-4: new_count_x-4 + rows, new_count_y-9: new_count_y-9 + cols]
#     #T2
#     # gt_array_ = gt_array[:, new_count_x-5: new_count_x-5 + rows, new_count_y-5: new_count_y-5 + cols]


#     T2_gt_1ch.extend(np.float32(gt_array_))


# T2_gt_1ch = np.asarray(T2_gt_1ch)
# T2_gt_1ch[T2_gt_1ch == 500] = 1
# T2_gt_1ch[T2_gt_1ch == 200] = 2
# T2_gt_1ch[T2_gt_1ch == 600] = 3
# sitk.WriteImage(sitk.GetImageFromArray(LGE_data_1ch),"/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_data_C0_224_224.nii.gz")
# np.save('/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_data_C0_224_224.npy', LGE_data_1ch[:, :, :, np.newaxis])

# sitk.WriteImage(sitk.GetImageFromArray(T2_gt_1ch),"/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_gt_C0_224_224.nii.gz")
# np.save('/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_gt_C0_224_224.npy', T2_gt_1ch)
# print(img_count)