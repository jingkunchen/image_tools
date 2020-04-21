import numpy as np
import cv2

orginal_path = "/Users/chenjingkun/Documents/data/COVID/xray/original"




    
    mask = np.zeros(np.shape(data_array), dtype='float32')
    thresh = 1
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0
    for iii in range(np.shape(data_array)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br
    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]

    data_array_ = ls
    [:,
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