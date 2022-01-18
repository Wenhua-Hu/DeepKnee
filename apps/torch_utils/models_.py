import os
import torch
import torch.nn as nn
from torch.serialization import SourceChangeWarning
from torchvision import models
import warnings

warnings.filterwarnings("ignore", category=SourceChangeWarning)
torch.nn.Module.dump_patches = True



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class multi_output_model(nn.Module):
    def __init__(self, model_core, num_ftrs):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.y1o = nn.Linear(num_ftrs, 2)
        self.y2o = nn.Linear(num_ftrs, 5)

    def forward(self, x):
        x1 = self.resnet_model(x)

        y1o = self.y1o(x1)
        y2o = self.y2o(x1)

        return y1o, y2o



def load_model(model_name, model_path, classes=5):
    name = model_name.lower()
    model = None
    if name.startswith('resnet'):
        if name == 'resnet18':
            model = models.resnet18(pretrained=False)
        elif name == 'resnet34':
            model = models.resnet34(pretrained=False)
        elif name == 'resnet50':
            model = models.resnet50(pretrained=False)
        elif name == 'resnet101':
            model = models.resnet101(pretrained=False)
        elif name == 'resnet152':
            model = models.resnet101(pretrained=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(model_path, map_location=device).state_dict(), strict=False)

    elif name.startswith('vgg'):
        if name == 'vgg16':
            model = models.vgg16_bn(pretrained=False)
        elif name == 'vgg19':
            model = models.vgg19_bn(pretrained=False)

        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(model_path, map_location=device).state_dict(), strict=False)

    elif name.startswith('inception'):
        if name == 'inception_v3':
            model = models.inception_v3(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, classes)
            model.load_state_dict(torch.load(model_path, map_location=device).state_dict(), strict=False)

    elif name.startswith('densenet'):
        if name == 'densenet121':
            model = models.densenet121(pretrained=False)
        elif name == 'densenet169':
            model = models.densenet169(pretrained=False)
        elif name == 'densenet201':
            model = models.densenet201(pretrained=False)

        num_ftrs = model.classifier.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(model_path, map_location=device).state_dict(), strict=False)

    model.name = name


    return model