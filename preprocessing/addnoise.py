import numpy
import cv2
import os
 

def addNoise(image,root,file):
    sr = numpy.arange(7,17,2)
    ir = numpy.arange(100,200,1)

    rand = numpy.random.randint(0, len(sr))
    size = sr[rand]
    half_size = (size - 1)/2
    lr = numpy.arange(half_size, 28-half_size, 1)
    rand = numpy.random.randint(0, len(lr))
    loc = lr[rand]

    noise = numpy.zeros(image.shape)
    rand = numpy.random.randint(0, len(ir), size*size)
    inten = ir[rand]
    inten = inten.reshape((size, size))
    noise[loc-half_size:loc+half_size+1, loc-half_size:loc+half_size+1] = inten
    noise_image = image + noise
    noise_image = numpy.clip(noise_image, 0, 255)
    print numpy.max(noise_image)
    #cv2.imwrite('/home/jingkunchen/data/mnist/testnoise/noise_'+file,noise_image)

for root, dirs, files in os.walk('/home/jingkunchen/data/mnist/test'):
    for file in files:
        if os.path.splitext(file)[1] == '.bmp':
            image = cv2.imread(os.path.join(root, file),cv2.IMREAD_GRAYSCALE)
            addNoise(image,root,file)
