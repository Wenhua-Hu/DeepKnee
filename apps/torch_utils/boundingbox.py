#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class BBoxerwGradCAM():

    def __init__(self, learner, heatmap, image_path, resize_scale_list, bbox_scale_list):
        self.learner = learner
        self.heatmap = heatmap
        self.image_path = image_path
        self.resize_list = resize_scale_list
        self.scale_list = bbox_scale_list

        self.og_img, self.smooth_heatmap = self.heatmap_smoothing()

        self.bbox_coords, self.poly_coords, self.grey_img, self.contours = self.form_bboxes()

    def heatmap_smoothing(self):
        og_img = cv2.imread(self.image_path)
        heatmap = cv2.resize(self.heatmap, (self.resize_list[0], self.resize_list[1]))  # Resizing
        og_img = cv2.resize(og_img, (self.resize_list[0], self.resize_list[1]))  # Resizing
        '''
        The minimum pixel value will be mapped to the minimum output value (alpha - 0)
        The maximum pixel value will be mapped to the maximum output value (beta - 155)
        Linear scaling is applied to everything in between.
        These values were chosen with trial and error using COLORMAP_JET to deliver the best pixel saturation for forming contours.
        '''
        heatmapshow = cv2.normalize(heatmap, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

        return og_img, heatmapshow


    def show_bboxrectangle(self):
        cv2.rectangle(self.og_img,
                      (self.bbox_coords[0], self.bbox_coords[1]),
                      (self.bbox_coords[0] + self.bbox_coords[2], self.bbox_coords[1] + self.bbox_coords[3]),
                      (0, 0, 0), 3)
        im = Image.fromarray(self.og_img)
        im.save("boundingbox.png")


    def form_bboxes(self):
        grey_img = cv2.cvtColor(self.smooth_heatmap, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey_img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        for item in range(len(contours)):
            cnt = contours[item]
            if len(cnt) > 20:
                # print(len(cnt))
                x, y, w, h = cv2.boundingRect(
                    cnt)  # x, y is the top left corner, and w, h are the width and height respectively
                poly_coords = [cnt]  # polygon coordinates are based on contours

                x = int(x * self.scale_list[0])  # rescaling the boundary box based on user input
                y = int(y * self.scale_list[1])
                w = int(w * self.scale_list[2])
                h = int(h * self.scale_list[3])

                return [x, y, w, h], poly_coords, grey_img, contours

            else:
                print("contour error (too small)")

    def get_bboxes(self):
        return self.bbox_coords, self.poly_coords


if __name__ == '__main__':
    from models_ import load_model
    model = load_model("resnet34", r"C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\data\models\resnet34_0.pth")

    image_resizing_scale = [400, 300]
    bbox_scaling = [1, 1, 1, 1]
    from PIL import Image
    image_path=r"C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\apps\static\assets\images\knee_gradcam\9003126L_gradcam.png"
    image = Image.open(image_path)
    gcam_heatmap = np.array(image)
    bbox = BBoxerwGradCAM(model,
                          gcam_heatmap,
                          image_path,
                          image_resizing_scale,
                          bbox_scaling)

    for function in dir(bbox)[-18:]:
        print(function)

    # bbox.show_smoothheatmap()
    # bbox.show_contouredheatmap()
    # bbox.show_bboxrectangle()
    # bbox.show_bboxpolygon()
    #
    bbox.show_bboxrectangle()
    # rect_coords, polygon_coords = bbox.get_bboxes()
