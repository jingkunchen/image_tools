import numpy
import scipy.misc
import os
import urllib
import gzip
import cPickle as pickle
import cv2
numpy.random.seed(1337)

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    print(numpy.array(images).shape)

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
        train_images, train_target = train_data
        dev_images, dev_target = dev_data
        train_images = numpy.concatenate((train_images, dev_images), axis=0)
        train_target = numpy.concatenate((train_target, dev_target), axis=0)

        train_data = (train_images, train_target)

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )

def load_noise_testing_images(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
        test_images, test_target = test_data

        test_images = add_square_noise(test_images)
        test_data = (test_images, test_target)

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )

def add_square_noise(train_images):

    # convert image back to [0, 255]
    train_images = train_images*255
    train_images = numpy.ceil(train_images)
    print train_images.shape[0]
    print train_images.shape[1]
    print train_images.shape[2]
    train_images = train_images.reshape((train_images.shape[0], 28, 28))
    
   # swapaxes
    # noise size range
    sr = numpy.arange(7,17,2)
    ir = numpy.arange(100,200,1)
    for i in range(train_images.shape[0]):
        image = train_images[i,:,:]
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
        noise_image = noise_image/numpy.max(noise_image)
        train_images[i,:,:] = noise_image
    train_images = train_images.reshape((train_images.shape[0], 28*28))

    return train_images


def bmp2jph():
    from PIL import Image
    pathDir = os.listdir(path)
    for i in pathDir:
        tmp = path+'/'+ i
        img = cv2.imread(tmp)
        img1 = add_square_noise(img)
        cv2.imwrite(path+'noise/'+i, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 
    img = Image.open('C:/Python27/image.bmp')
    new_img = img.resize( (28, 28) )
    new_img.save( 'C:/Python27/image.png', 'png')

def addNoiseHandler(path):
    pathDir = os.listdir(path)
    for i in pathDir:
        tmp = path+'/'+ i
        img = cv2.imread(tmp)
        img1 = add_square_noise(img)
        cv2.imwrite(path+'noise/'+i, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 

def main():
    addNoiseHandler("/home/jingkunchen/data/mnist/test")

if __name__ == "__main__":
    main()