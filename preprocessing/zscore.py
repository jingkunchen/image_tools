import numpy as np
import cv2
import random
import os
import glob
from PIL import Image
from torchvision import transforms

transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)
 
# calculate means and std
testPath = '/Users/chenjingkun/Documents/data/COVID/in/'
means = 0
stdevs = 0
fileNames = os.listdir(testPath)
img_list = []
pixel = []
num_imgs = 0
for i in fileNames:
    print(i)
    num_imgs += 1
    #,, cv2.IMREAD_GRAYSCALE
    img = cv2.imread(testPath+i, cv2.IMREAD_GRAYSCALE)
    img1 = transform1(img)
    img2 = img1.numpy()
    print(img2.shape)
    # img2 = img2.astype(np.float32)
    print("img.mean():",img2.mean())
    means += img2.mean()
    print("means:",means)
    stdevs += img2.std()
    print("stdevs:",stdevs)

# means.reverse()
# stdevs.reverse()

means = means / num_imgs
stdevs = stdevs / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))




    # print(i)
    # with Image.open(testPath+ i) as img:
    #     # img.show()
    #     # rgb_im = img
    #     rgb_im = img.convert('RGB')
    #     print(rgb_im)
    #     for i in range(1024):
    #         for j in range(1024):
    #             r, g, b = rgb_im.getpixel((i, j))
    #             pixel.append(r)

    # print(img.shape)
    # img = cv2.resize(img, (512,512))
    # print(img.shape)
        
# img_array = np.array(img)

# pixel= np.array(pixel)

# print(np.mean(pixel))
# print(np.std(pixel))
# index = 1
# num_imgs = 0
# with open(train_txt_path, 'r') as f:
#     lines = f.readlines()
#     random.shuffle(lines)
#     # lines = lines[:2]
 
#     for line in lines:
#         eles = line.strip().split(' ')
#         print('{}/{}'.format(index, len(lines)))
#         index += 1
 
#         datas = glob.glob(os.path.join(eles[0], 'diff_nor*.jpg'))
#         for data in datas:
#             num_imgs += 1
#             img = cv2.imread(data)
#             img = img.astype(np.float32).
#             for i in range(3):
#                 means[i] += img[:, :, i].mean()
#                 stdevs[i] += img[:, :, i].std()
 
# means.reverse()
# stdevs.reverse()
 
# means = np.asarray(means) / num_imgs
# stdevs = np.asarray(stdevs) / num_imgs
 
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
# print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
