import torch.nn.functional as F
import torch
import torch.utils.data as tdata
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function
from PIL import Image
import os
import cv2
import care_model
import numpy as np


class GradModel(nn.Module):
    """In order to save gradients during training, modifying model here.
       Wrap pretrained-model, make features function and classifier
     """
    def __init__(self, original_model, num_classes=7):
        super(GradModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[: -2])
        self.avgpool = original_model.avgpool     
        self.classifier = original_model.fc
        self.gradients = []
        self.target_layers = ["7"]  # for resnet50

    def get_gradients(self):
        """Get gradients.
        """
        return self.gradients

    def extractor(self, inputs):
        """extract features and outputs
        Args:
            inputs: images [batch_size, channel, height, width]
        Returns:
            features:
            outputs:
        """
        def save_gradient(grad):
            """save_gradient"""
            self.gradients.append(grad)

        features = []
        self.gradients = []

        # print("Input size: {}".format(inputs.size())
        outputs = inputs
        for name, module in self.features._modules.items():
            outputs = module(outputs)
            if name in self.target_layers:
                outputs.register_hook(save_gradient)
                features += [outputs]

        outputs = self.avgpool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        y_pred = self.classifier(outputs)
        return features, y_pred

    def forward(self, inputs):
        out = self.features(inputs)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(F.relu(out))
        return out

def network(pretrained=None):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 3)
    model = GradModel(model, 3)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    return model


class GradCam:
    def __init__(self, net, cls_num, target_cls, use_origin_cam= False):
        self.net = net
        self.save_path = args.save_path
        self.use_cuda = args.use_cuda
        self.cls_num = cls_num
        self.target_cls = target_cls
        self.create_directory()

        self.batch_size = 32

        # chek_point = torch.load(args.checkpoint)
        # d = chek_point['state_dict']
        # self.net.load_state_dict(d)

        self.use_origin_cam = use_origin_cam

    # def origin_cam(self, features, weights):


    def grad_cam(self, target_activation, grad_val):
        """grad for each image
        Args:
            target_activation: [2048, 7, 7]
            grad_val: [2048, 7, 7]
        Return:
            grad: [224,224]
        """
        # [2048, 7, 7]
        channels, h, w = grad_val.size()
        # [2048, 49]
        grad_val = grad_val.view(channels, -1)
        weights = torch.mean(grad_val, dim=1)
        # print(weights.size())
        # [2048]
        weights = weights.view(channels, 1, 1)
        # print(weights.size())
        cam = weights * target_activation

        # [2048]
        cam = torch.sum(cam, dim=0)
        # print(cam.size())
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        cam = cam.view(1, 1, h, w)
        # resize_img = torch.nn.Upsample(size=(224,224), mode='bilinear')
        cam = torch.nn.functional.interpolate(cam, size=(512, 512), mode='bilinear')
        return cam.squeeze()


    def save_image(self, input, cam, dir, img_name, mask=None):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = input

        img = img.permute(1, 2, 0).cpu().data

        # img = np.zeros((224, 224, 3))
        # c, h, w = input.shape
        # for k in range(c):
        #     for j in range(h):
        #         for i in range(w):
        #             img[j, i, k] = input[k, j, i]
        # if mask is not None:
        #     img[np.where(mask != 0)] += [0.3, 0.3, 0]

        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        # img = np.uint8(255 * img)
        # numpy_horizontal_concat = np.concatenate((img, cam), axis=1)
        path = os.path.join(dir, img_name)
        cv2.imwrite(path, cam)



    def get_cams(self, train=False, num_select=-1, fold=0):
        self.net.eval()
        index = 0
        data_root = 'E:/Covid/covid_dataset/{}/covid/'.format(fold)
        totensor = transforms.ToTensor()
        mean = 0.4953
        std = 0.2527
        for f in os.listdir(data_root):
            img = Image.open(data_root + f).convert('L').resize((512, 512), resample=Image.BILINEAR)
            x = totensor(img)
            x = (x - mean) / std
            x = x.to(device)
            x = x.unsqueeze(0)
            x = x.repeat(1, 3, 1, 1)
            targets = torch.Tensor([self.target_cls]).to(device, torch.long)

            feature, output = self.net.extractor(x)
            targets_onehot = F.one_hot(targets, 3).to(torch.float)
            targets_value = targets_onehot * output
            sum_output = torch.sum(targets_value)
            sum_output.backward()

            grad_val = self.net.get_gradients()[-1]
            pred = output.argmax(dim=1)
            feature = feature[0]
            grad_val = grad_val[0]

            cam = self.grad_cam(feature[0], grad_val).cpu().data.numpy()
            name = '{}_{}_{}.jpg'.format(f[:-4], targets[0], pred[0])

            self.save_image(x.squeeze() * std + mean, cam, self.save_path, name)
            # print('{}: {}/{}'.format(index, path, name))
            if num_select > 0 and index >= num_select:
                return None
            index += 1

if __name__ == '__main__':
    # if args.use_cuda:
    #     print("Using GPU for acceleration")
    # else:
    #     print("Using CPU for computation")
    device = torch.device('cuda')

    fold = 0
    net = network('models/care_fold{}_alpha=0.5.pth'.format(fold))
    net.to(device)
    gradcam = GradCam(net, target_cls=2)
    gradcam.get_cams(train=False, num_select=100, fold=0 if fold else 1)