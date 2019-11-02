import cv2
import os

def  getFiles(file_path):
    #a = os.walk(file_path)
    file_list = []
    
    for root, dirs, files in os.walk(file_path):  
        for file in files: 
            if os.path.splitext(file)[1] == '.jpg':  
                file_list.append(os.path.join(root, file))
    return file_list

def showImage(img):
    cv2.imshow('src',img)
    cv2.waitKey(0)

def readImage(file_path):
    img = cv2.imread(file_path)

    
    return img

def cropImage(img, new_width, new_height, d):
    img = img[0:new_width, 0:new_height, 0:d]   
    print(img.shape)
    return img

if __name__ == "__main__":
    file_path = '/home/chenjingkun/Documents/data/uiuc_texture_dataset/T01'
    file_list = getFiles(file_path)
    print(file_list)
    for i in file_list:
        showImage(cropImage(readImage(i),480,480,1))