import os
import random
rootPath = "/home/jingkunchen/AnoGAN/data/ISBI2016_ISIC_All_Clean_patch_128_160/"
healthPath = "/home/jingkunchen/AnoGAN/data/health/"
healthTrain = "/home/jingkunchen/AnoGAN/data/healthtrain/"
lensionPath = "/home/jingkunchen/AnoGAN/data/lension/"
lensionTest = "/home/jingkunchen/AnoGAN/data/lensiontest/"

brainRootPath = "/home/jingkunchen/data/brain/train/"
brainHealthPath = "/home/jingkunchen/data/brain/train/"
brainlesionPath = "/home/jingkunchen/data/brain/test/"
def cpPicture(srcDir, dstDir, tag):
    for file in os.listdir(srcDir): 
        filePath = srcDir+file+'/'+tag
        cmd = 'cp '+ filePath +'/* '+ dstDir
        os.system(cmd)

def calcNumber(Dir):
    return len([name for name in os.listdir(Dir) if os.path.isfile(os.path.join(Dir,name))]) 
    '''
    count = 0
    ls = os.listdir(Dir)
    for i in ls:
        if os.path.isfile(os.path.join(Dir,i)):
            count += 1
    print count
    '''

def mvPicture(srcDir, dstDir):
    pathDir = os.listdir(srcDir)
    pathDir1 = []
    for i in pathDir:
        pathDir1.append( i)
    #print pathDir1
    count = int(calcNumber(srcDir))*0.2
    #print count
    sample = random.sample(pathDir, int(count))
    #print sample
	
    for name in pathDir1:
        cmd = 'cp '+ " /home/jingkunchen/data/brain/test/" + name + '/health/*' + ' ' +'/home/jingkunchen/data/brain/testimg/health/' 
        print cmd
        os.system(cmd)

def main():
   #cpPicture(rootPath,healthPath, 'health') 
   #cpPicture(rootPath,lensionPath, 'lension') 
   #mvPicture(healthPath, healthTrain)
   #mvPicture(lensionPath, lensionTest)
   mvPicture("/home/jingkunchen/data/brain/test/","/home/jingkunchen/data/brain/testimg/")

if __name__ == "__main__":
    main()
