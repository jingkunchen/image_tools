import os
import cv2

#path = "/home/jingkunchen/test/2/AnoGAN/data/skin_64_64/"
#path1 = "/home/jingkunchen/test/2/AnoGAN/data/skin_32_32/"

path = '/home/jingkunchen/data/skin_64_64/lesiontest/'
path1 = '/home/jingkunchen/data/skin_32_32/lesiontest/'

#path = "/home/jingkunchen/anogan-tf/test_data/original/"
#path1 = "/home/jingkunchen/anogan-tf/test_data/"


files= os.listdir(path)

for file in files:
    pic = cv2.imread(path +file)
    print file
    pic1 = cv2.resize(pic, (32,32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path1+file, pic1)
