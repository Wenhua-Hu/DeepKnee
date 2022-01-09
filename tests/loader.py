# -*- coding: utf-8 -*-

import os

import torch
import torchvision.models as models
from torchvision import transforms

from tests.knee_sets import ImageFolder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def data_load():
    pixel_mean, pixel_std = 0.66133188, 0.21229856
    phases = ['train', 'val', 'test', 'auto_test']
    # phases = ['train', 'val', 'test', 'auto_test']
    data_transform = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
        ]),
        'auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
        ]),
        'most_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
        ]),
        'most_auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
        ])
    }

    dsets = {x: ImageFolder(
        os.path.join(r"C:\Users\whu\Desktop\sci_repo\UvA\DSP\DeepKnee\apps\static\assets\images", "knee"),
        data_transform[x]) for x in phases}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=1,
                                                   shuffle=False, num_workers=0) for x in phases}

    dset_size = {x: 6 for x in phases}

    return dset_loaders, dset_size, 5


class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, num_ftrs):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        # heads
        self.y1o = nn.Linear(num_ftrs, 2)
        self.y2o = nn.Linear(num_ftrs, 5)

    def forward(self, x):
        x1 = self.resnet_model(x)

        ## only get until the FC

        # heads
        y1o = self.y1o(x1)
        y2o = self.y2o(x1)

        return y1o, y2o


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.autograd import Variable


    def eval_model(phase, dset_loaders, dset_size):
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential()
        model = multi_output_model(model, num_ftrs)
        model.load_state_dict(torch.load("./best_weights.pth", map_location=torch.device('cpu')))
        # print(model)

        model.eval()

        labels_all = [] * dset_size[phase]
        preds_all = [] * dset_size[phase]
        feas_all = []

        for data in dset_loaders['test']:
            inputs, labels, paths = data

            inputs = Variable(inputs)
            print(inputs.shape)
            with torch.no_grad():
                outputs = model(inputs)

                a = torch.argmax(outputs[0], 1)
                b = torch.argmax(outputs[1], 1)
                print(outputs[0], outputs[1])
                print(a, b)


    a, b, c = data_load()
    eval_model('test', a, b)
