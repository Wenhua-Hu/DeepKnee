# -*- coding: utf-8 -*-
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CalLime(object):

    def __init__(self, model, img_path, outdir, num_samples):
        self.model = model
        self.img_path = img_path
        self.outdir = outdir
        self.num_samples = num_samples
        self.pil_img = self.image_preprocess()
        self.pil_transform = self.get_pil_transform(299)
        self.preprocess_transform = self.get_preprocess_transform()

    def image_preprocess(self):
        img = cv2.imread(self.img_path, 1)
        return Image.fromarray(img, 'RGB')

    def get_pil_transform(self, size):
        '''resize function'''
        transf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size)
        ])
        return transf

    def get_preprocess_transform(self):
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.66133188] * 3, [0.21229856] * 3)
        ])

        return transf

    def batch_predict(self, images):
        '''
        Predictive function. Lime requires the function to only have one input
        In this case an array of images (usually with only on element), these are
        preprocessed and evaluated

        '''
        self.model.eval()

        # To not change the img_preprocesses function the following line is somewhat
        # arbritrary, but the point is to transofrm the input image to a numpy array
        # because img_preprocess requires that. Then it squeezed back the added
        # dimensions because LIME requires that.
        batch = torch.stack(tuple(np.squeeze(self.preprocess_transform(np.array(i))) for i in images), dim=0)

        # basic prediction
        self.model.to(device)
        batch = batch.to(device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def lime_predict(self):
        # initiate explainer
        explainer = lime_image.LimeImageExplainer()

        # run the explanation
        explanation = explainer.explain_instance(np.array(self.pil_img),
                                                 self.batch_predict,
                                                 top_labels=1,
                                                 hide_color=0,
                                                 num_samples=self.num_samples)

        # get overlay
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=False)
        img_boundary = mark_boundaries(temp / 255.0, mask)
        filename = os.path.basename(self.img_path)
        name = os.path.splitext(filename)[0]
        plt.imsave(os.path.join(self.outdir, name + '_lime.png'), img_boundary)


def lime_run(model, img_path, IMAGES_KNEE_LIME, num_samples):
    clime = CalLime(model, img_path, IMAGES_KNEE_LIME, num_samples)
    clime.lime_predict()
