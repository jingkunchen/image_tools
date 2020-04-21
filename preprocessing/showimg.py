import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_in_one(images, show_size=(32*4+10, 32*4+10), blank_size=2, window_name="health"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("图片总数为： %s" % (max_count - count))
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, merge_img)
    cv2.imwrite("brain_lesion.png", merge_img*255.)
    # cv2.waitKey(0)
    # cv2.destroyWindow()


# tmp = cv2.imread("ISIC_0000483_1_129.jpg")
# tmp = tmp/255.
image_health = np.load("covid_original.npy")
# image_health = np.squeeze(image_health, axis=(3,))
# image_health = normalize(image_health)
# image_lesion = np.load("covid_original.npy")
# image_lesion = np.squeeze(image_lesion, axis=(3,))
print("image_health:",image_health.shape)
# print("image_lesion:",image_lesion.shape)
list1 = []
for i in range(0,16):

    print(image_health[i])
    # cv2.imwrite(str(i)+"health.png",imag de_health[i][:16,:16,:]*255.,[int( cv2.IMWRITE_JPEG_QUALITY), 95])
    # cv2.imwrite(str(i)+"lesion.png",image_lesion[i]*255.,[int( cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imshow("xray",image_health[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # fig= plt.figure(figsize=(8, 8))

# show_in_one(list1)
    # columns = 1
    # rows = 2
    # fig.add_subplot(rows, columns, 1)
    # plt.title('health')
    # plt.imshow(image_health[i], label='health')
    # fig.add_subplot(rows, columns, 2)
    # plt.title('lesion')
    # plt.imshow(image_lesion[i], label='lesion')
    # plt.show()