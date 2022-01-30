import os
import warnings
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fmap_block = list()
grad_block = list()


def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.53995493] * 3, [0.27281794] * 3)
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)

def cam_show_img(img, feature_map, grads, out_image_location):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0], -1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_image_location)
    cv2.imwrite(path_cam_img, cam_img)


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

