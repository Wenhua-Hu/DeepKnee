import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiOutputModel(nn.Module):
    def __init__(self, model, num_features):
        super(MultiOutputModel, self).__init__()

        self.model = model
        self.y1o = nn.Linear(num_features, 2)
        self.y2o = nn.Linear(num_features, 5)

    def forward(self, x):
        x1 = self.model(x)

        y1o = self.y1o(x1)
        y2o = self.y2o(x1)

        return y1o, y2o


def data_load():
    pixel_mean, pixel_std = 0.66133188, 0.21229856
    phases = ['train', 'val', 'test']
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
        ])
    }

    knee_dir = os.path.join(r"C:\Users\whu\Desktop\sci_repo\UvA\DSP\DeepKnee\apps\static\assets\images", "knee")
    dsets = {x: ImageFolder(knee_dir, data_transform[x]) for x in phases}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=1, shuffle=False, num_workers=0) for x in
                    phases}

    dset_size = {x: 6 for x in phases}

    return dset_loaders, dset_size, 5


def eval_model(phase, dset_loaders, dset_size):
    model = models.resnet34(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential()
    model = multi_output_model(model, num_ftrs)
    model.load_state_dict(torch.load("./best_weights.pth", map_location=torch.device('cpu')))

    model.eval()

    labels_all = [] * dset_size[phase]
    preds_all = [] * dset_size[phase]
    feas_all = []

    # for data in dset_loaders['test']:
    # inputs, labels, paths = data
    #
    # inputs = Variable(inputs)
    #
    #
    #
    # with torch.no_grad():
    #     outputs = model(inputs)
    #
    #     a = torch.argmax(outputs[0],1)
    #     b = torch.argmax(outputs[1], 1)
    #     print(outputs[0], outputs[1])
    #     print(a, b)

    get_predictions(model, dset_loaders['test'])


def get_predictions(model, iterator):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            try:
                y_pred, _ = model(x)
            except:
                y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


if __name__ == '__main__':
    a, b, c = data_load()
    eval_model('test', a, b)
