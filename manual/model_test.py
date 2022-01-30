import copy

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
a = img_preprocess(r"C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\static\assets\images\knee_boundingbox\9001897L_boundingbox.png")
model = models.resnet34(pretrained=False)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 5)
model_path = r'C:\Users\whu\Downloads\0.687-0.37.pth'
print(torch.load(model_path, map_location=device))
# model.load_state_dict(torch.load(model_path, map_location=device).state_dict(), strict=False)

# torch.nn.Module.dump_patches = True
# print(model(a))

# for i, module in model._modules.items():
#     a = module(a)










