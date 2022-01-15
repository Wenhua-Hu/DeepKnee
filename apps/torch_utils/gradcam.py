import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class cal_cam(nn.Module):
    def __init__(self, model, outdir, feature_layer):
        super(cal_cam, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)

        self.feature_layer = feature_layer
        self.gradient = []
        # record the feature map
        self.output = []
        self.means = [0.53995493] * 3
        self.stds = [0.27281794] * 3
        self.outdir = outdir

        self.transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(self.means, self.stds)
        ])

    def save_grad(self, grad):
        self.gradient.append(grad)

    def get_grad(self):
        return self.gradient[-1].cpu().data

    def get_feature(self):
        return self.output[-1][0]

    def process_img(self, input):
        input = self.transform(input)
        input = input.unsqueeze(0)
        return input

    def getGrad(self, input_):
        input_ = input_.to(self.device).requires_grad_(True)
        input = self.forward(input_)

        index = torch.max(input, dim=-1)[1]
        one_hot = torch.zeros((1, input.shape[-1]), dtype=torch.float32)
        one_hot[0][index] = 1
        confidenct = one_hot * input.cpu()
        confidenct = torch.sum(confidenct, dim=-1).requires_grad_(True)
        # print(confidenct)
        self.model.zero_grad()
        # backpropogate to get th gredient
        confidenct.backward(retain_graph=True)
        # get the gradient of feature map
        grad_val = self.get_grad()
        feature = self.get_feature()
        return grad_val, feature, input_.grad

    # 计算CAM
    def getCam(self, grad_val, feature):
        # GVP for per channel
        alpha = torch.mean(grad_val, dim=(2, 3)).cpu()
        feature = feature.cpu()

        cam = torch.zeros((feature.shape[2], feature.shape[3]), dtype=torch.float32)
        for idx in range(alpha.shape[1]):
            cam = cam + alpha[0][idx] * feature[0][idx]
        # ReLu
        cam = np.maximum(cam.detach().numpy(), 0)

        plt.imshow(cam)
        plt.colorbar()
        plt.savefig(os.path.join(self.outdir,self.name + '_gradcam_0.png'))

        cam_ = cv2.resize(cam, (299, 299))
        cam_ = cam_ - np.min(cam_)
        cam_ = cam_ / np.max(cam_)
        plt.imshow(cam_)
        plt.savefig(os.path.join(self.outdir,self.name + '_gradcam_1.png'))
        cam = torch.from_numpy(cam)

        return cam, cam_

    def show_img(self, cam_, img):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_), cv2.COLORMAP_JET)

        cam_img = 0.3 * heatmap + 0.7 * np.float32(img)
        cv2.imwrite(os.path.join(self.outdir,self.name + '_gradcam.png'), cam_img)

    def __call__(self, img_root):
        filename = os.path.basename(img_root)
        self.name = os.path.splitext(filename)[0]
        img = Image.open(img_root)
        img = img.resize((299, 299))
        img = img.convert("RGB")

        input = self.process_img(img)
        grad_val, feature, input_grad = self.getGrad(input)
        cam, cam_ = self.getCam(grad_val, feature)
        self.show_img(cam_, img)
        return cam


class gradcam_resnet(cal_cam):

    def __init__(self, model, outdir):
        super(gradcam_resnet, self).__init__(model, outdir, "layer4")

    def forward(self, input_):
        num = 1
        for name, module in self.model._modules.items():
            if name == 'resnet_model':
                for name, module in module._modules.items():
                    if (num == 1):
                        input = module(input_)
                        num = num + 1
                        continue

                    if (name == self.feature_layer):
                        input = module(input)
                        input.register_hook(self.save_grad)
                        self.output.append([input])

                    elif (name == "avgpool"):
                        input = module(input)
                        input = input.reshape(input.shape[0], -1)

                    else:
                        input = module(input)

            elif name == 'y2o':
                input = module(input)

        return input

