import random
import os
path_0 = '/home/jingkun/data/Cell_Zhangkang/covid_orginal/0/CP' #NCP,Normal
path_1 = '/home/jingkun/data/Cell_Zhangkang/covid_orginal/1/CP'
dst = '/home/jingkun/data/Cell_Zhangkang/covid_orginal/CP'

imgs_path = []
dst_path = []

for x in os.listdir(path_0):
    if x.endswith(".jpg"):
        imgs_path.append(os.path.join(path_0,x))
        dst_path.append(os.path.join(dst,x))

for x in os.listdir(path_1):
    if x.endswith(".jpg"):
        imgs_path.append(os.path.join(path_1,x))
        dst_path.append(os.path.join(dst,x))

selected_imgs = random.sample(list(range(len(imgs_path))), k=50000)

for i in range(50000):
    os.system("cp "+imgs_path[i]+" "+dst_path[i])
print("done")

