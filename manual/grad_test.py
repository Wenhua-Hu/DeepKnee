# -*- coding: utf-8 -*-

import os, sys, pdb

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import numpy as np
import cv2


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        # self.model.eval()
        self.cuda = 0

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        one_hot = torch.sum(one_hot.cuda(0) * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.ones(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

if __name__ == '__main__':
    import cv2
    from torch.serialization import SourceChangeWarning
    from torchvision import models
    import torch.nn as nn
    import torch
    import warnings
    import numpy as np
    from torchvision.transforms import transforms

    warnings.filterwarnings("ignore", category=SourceChangeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # model = models.vgg16_bn(pretrained=False)
    # num_ftrs = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_ftrs, 5)

    def img_preprocess(img_path):
        img_in = cv2.imread(img_path, 1)
        img = img_in.copy()
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.66133188] * 3, [0.21229856] * 3)
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        return img


    a = img_preprocess(
        r"C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\static\assets\images\knee_boundingbox\9001897L_boundingbox.png")
    model = models.densenet121(pretrained=False)

    num_ftrs = model.classifier.in_features

    model.classifier = nn.Linear(num_ftrs, 5)
    model_path = r'C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\data\models\densenet121_1.pth'
    model.load_state_dict(torch.load(model_path, map_location=device).state_dict(), strict=False)

    mo = GradCam(model, 'features')
    b = mo(a)
    # print(b)
