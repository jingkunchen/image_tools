import os
import scipy
import utils
import numpy as np
srcDir= '/home/jingkunchen/data/brain/brainset/trainimg/health/' 
def imread(path, grayscale = False):
    if (grayscale):
            return scipy.misc.imread(path, flatten = True).astype(np.int)
    else:
            return scipy.misc.imread(path)
images = []
pathDirs = []
pathDir = os.listdir(srcDir)
for i in pathDir[0:100]:
    pathDirs.append(srcDir + i)
for name in pathDirs[0:100]:
    image_tmp = imread(name)
    #image_tmp = scipy.misc.imresize(image_org.astype(float), [32, 32])
    image = np.array(image_tmp)/127.5 - 1.
    images.append(image)

tmp = np.array(images)[:,:,:,None]
utils.save_images(tmp.astype(float), utils.image_manifold_size(tmp.shape[0]),'1.png')

