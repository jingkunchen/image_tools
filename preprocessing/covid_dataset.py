import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils import data

class CovDataset(data.Dataset):
    def __init__(self, root, fold, transform=None):
        super(CovDataset, self).__init__()
        self.transform = transform if transform is not None else lambda x: x
        self.images = []
        self.labels = []
        root = os.path.join(root, 'covid_dataset_validation', str(fold))

        for f in os.listdir(os.path.join(root, 'normal')):
            self.labels.append(0)
            self.images.append(os.path.join(root, 'normal', f))
        
        for f in os.listdir(os.path.join(root, 'pneumonia')):
            self.labels.append(1)
            self.images.append(os.path.join(root, 'pneumonia', f))
        
        for f in os.listdir(os.path.join(root, 'covid')):
            self.labels.append(2)
            self.images.append(os.path.join(root, 'covid', f))

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)
        label = self.labels[index]
        img_array=img.load()
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.labels)

def get_dataloader(bs, fold, nw, train):
    if not train:
        t = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            #transforms.Normalize((0.2354,), (0.2767,)),
            transforms.Normalize((0.4953,), (0.2527,)),
        ])
    else:
        '''
        t = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(30, translate=(0.05, 0.05), scale=(0.8, 1.2), resample=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.4953,), (0.2527,)),
        ])
        '''
        t = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(30, translate=None, scale=(0.8, 1.2), resample=Image.BILINEAR),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize((0.2354,), (0.2767,)),
            transforms.Normalize((0.4953,), (0.2527,)),
        ])
    
    ds = CovDataset('', fold, t)
    dataloader = data.DataLoader(ds, bs, True, num_workers=nw, pin_memory=True, drop_last=train)

    return dataloader

if __name__ == '__main__':
    #from covid_dataset import get_dataloader
    #train_loader = get_dataloader(16, train_fold, 4, True)
    ds = CovDataset('', 0, transforms.ToTensor())
    loader = data.DataLoader(ds, 64)
    
    for (x, label) in loader:
        print(label)
