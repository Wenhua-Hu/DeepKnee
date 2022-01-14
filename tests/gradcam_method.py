import os
import warnings
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from apps.home._models import multi_output_model, get_model

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
