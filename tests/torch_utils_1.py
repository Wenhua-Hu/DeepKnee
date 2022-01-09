import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from tests.knee_sets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.L1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.L1(x)
        out = self.relu(out)
        return self.L2(out)


class multi_output_model(nn.Module):
    def __init__(self, model_core, num_ftrs):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.y1o = nn.Linear(num_ftrs, 2)
        self.y2o = nn.Linear(num_ftrs, 5)

    def forward(self, x):
        x1 = self.resnet_model(x)

        # y1o = self.y1o(x1)
        y2o = self.y2o(x1)

        return y2o


def data_load():
    pixel_mean, pixel_std = 0.53995493, 0.27281794
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

    knee_dir = os.path.join(r"/apps/static/assets/images", "knee")
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


if __name__ == '__main__':
    from pytorch_grad_cam.utils.image import show_cam_on_image

    import os
    import torch
    import torchvision.models as models
    from torchvision import transforms

    from gradcam import GradCAM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.resnet34(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential()
    model = multi_output_model(model, num_features)
    model.load_state_dict(torch.load("./best_weights.pth", map_location=torch.device('cpu')))

    print(model)
    print(model.resnet_model.layer4[-1])
    # for name, layer in model.named_modules():
    #     print(name, layer)
    target_layer = model.resnet_model.layer4[-1].conv2

    model.eval()
    #
    filename = '9001400L.png'
    IMAGE_PATH = r"C:\Users\whu\Desktop\sci_repo\UvA\DSP\DeepKnee\apps\static\assets\images"
    knee_image = os.path.join(IMAGE_PATH, 'knee', filename)
    img = Image.open(knee_image)
    img = img.convert('RGB')

    test_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.53995493] * 3, [0.27281794] * 3)])

    img_tensor = test_transforms(img).to(device).unsqueeze(0)
    #
    #
    cam = GradCAM(model, target_layer=target_layer)
    grayscale_cam = cam(img_tensor)
    outputs = model(img_tensor)
    # print(grayscale_cam)
    grayscale_cam = grayscale_cam[0]
    print(torch.squeeze(grayscale_cam))

    test_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.53995493] * 3, [0.27281794] * 3)])

    visualization = show_cam_on_image(test_transforms(img), grayscale_cam)
    # y_proba_1 = F.softmax(outputs[0], dim=-1)
    # y_proba_2 = F.softmax(outputs[1], dim=-1)
    #
    # print(y_proba_1, y_proba_2)
    # print(outputs[0], outputs[1])

    # a, b,c = data_load()
    # eval_model('test', a, b)
