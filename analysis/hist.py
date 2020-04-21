import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


import cv2
import SimpleITK as sitk
import skimage.io as io
 
def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data
#显示一个系列图
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i,:,:], cmap = 'gray')
        print(i)
        io.show()
 
#单张显示
def show_img(ori_img):
    io.imshow(ori_img[150], cmap = 'gray')
    io.show()
 



path = '/Users/chenjingkun/Documents/data/stacom/c0/patient12_C0.nii.gz' #数据所在路径
data = read_img(path)
print(data.shape)
vals = data[8].flatten()
list_1 = []
for i in vals:
    
    if i !=0.0:
        print(i)
        list_1.append(i)

plt.hist(list_1, 400, [0,2300])
plt.title('')
plt.show()
# show_img(data)

# # example data
# mu = 100 # mean of distribution
# sigma = 15 # standard deviation of distribution
# x = mu + sigma * np.random.randn(10000)

# num_bins = 20
# the histogram of the data
# n, bins, patches = plt.hist(data[150], num_bins, normed=1, facecolor='blue', alpha=0.5)
# num_bins = 50
# hist = cv2.calcHist([data[150]], [0], None, [2500], [-1500, 1000])
# n, bins, patches = plt.hist(data[150], num_bins, normed=1, facecolor='blue', alpha=0.5)
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([-1000, 2500])
# plt.show()

# plt.plot(bins, 'r--')  
# plt.xlabel('Smarts')  
# plt.ylabel('Probability')  
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')  
    
# # Tweak spacing to prevent clipping of ylabel  
# plt.subplots_adjust(left=0.15)  
# plt.show()  