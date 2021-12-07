#try:
from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import pandas as pd
from PIL import Image
import streamlit as st
import os

from modules.grad_cam import GradCAM
from modules.ground_matrix import Groupmatrix, HeatmapSquares

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import cv2
#import imutils
import matplotlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dropout, Input, Lambda, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, Adamax
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import imagenet_utils

from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.python.keras.applications.efficientnet import EfficientNetB6, preprocess_input

from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
#except Exception as e:
#    print(e)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
enabled = False



st.set_page_config(
    page_title="DeepKnee Platform",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Quin & UvA"
    })

def upload_image():
    uploaded_image = st.file_uploader("Upload image:", type=["png", "jpg", 'jpeg'])
    if uploaded_image is not None:
        st.write("Original image: " + uploaded_image.name)
        image_file = load_image(uploaded_image)
        st.image(image_file, width=700)
        with open(os.path.join(BASE_DIR, "data/uploads", uploaded_image.name),"wb") as f:

            f.write(uploaded_image.getbuffer())
            enabled = True
            uploaded_image.url = os.path.join(BASE_DIR, "data/uploads", uploaded_image.name)
    return uploaded_image

def sider_bar():
    mod_list = ["ResNet", "VGG", "AlexNet"]
    container = st.sidebar.container()
    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)
    all = st.sidebar.checkbox("All Models", value=False)
    if all:
        selected_options = container.multiselect("Models options:", mod_list, mod_list)
    else:
        selected_options = container.multiselect("Models options:", mod_list)
    return container

@st.cache
def load_image(image_file):
    image = Image.open(image_file)
    return image

def show_prediction1(image_file, model):
    col1, col2 = st.columns(2)
    original = Image.open(image_file)
    col1.header("Original:")
    col1.image(original, use_column_width=True)

    grayscale = original.convert('LA')
    col2.header("Predicted:")
    col2.image(grayscale, use_column_width=True)
    st.write("Confidence Score: ")





def load_image2(path_to_image):
    orig = cv2.imread(path_to_image)
    resized = cv2.resize(orig, (224, 224))

    image = load_img(path_to_image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return orig, imagenet_utils.preprocess_input(image)


def predict_on_image(IMG_PATH, model):

    orig = cv2.imread(IMG_PATH)
    resized = cv2.resize(orig, (224, 224))

    image = load_img(IMG_PATH, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    labels = [0,1,2,3,4]
    preds = model.predict(image)
    i = np.argmax(preds[0])
    label = labels[i]
    percentage = preds[0][np.argmax(preds)]*100
    return (label, percentage)

def get_heatmap(model, image, label, orig):
    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, label)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    return cam.overlay_heatmap(heatmap, orig, alpha=0.5)


def show_prediction(obj_image, model):
    #path to example image
    IMG_PATH = obj_image.url #r'C:\Users\whu\Desktop\DSP_project\UVA21_DSP_QUIN\materials\Lukas\9999862R.png'
    #path to stored model: /VGG16-acc49 (whole folder with .pb file in it)
    MODEL_PATH = r'data\models\VGG16-acc49'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])
#LOAD MODEL:
    model = keras.models.load_model(MODEL_PATH)

    #load resized and original image
    orig, image = load_image2(IMG_PATH)

    #get model prediction and percentage
    label, percentage = predict_on_image(IMG_PATH, model)

    #get heatmap picture and heatmap picture on top of original
    heatmap, output = get_heatmap(model, image, label, orig)


    squares = draw_squares(model, label, image, orig, intensity_based_size=True,accuracy=percentage)


    col1, col2 = st.columns(2)
    # original = Image.open(MODEL_PATH)
    col1.header("Heatmap:")
    col1.image(heatmap, use_column_width=True)

    # grayscale = original.convert('LA')
    col2.header("Overlap:")
    # col2.subheader("Label:",label ,"Confidence Score: ", percentage)
    col2.image(output, use_column_width=True)
    col3, col4 = st.columns(2)
    col3.header("Bounding Box:")
    st.text("")
    col3.image(squares, use_column_width=True)
    #plt.figure(figsize = (15,15))
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    #plt.imshow(squares)
    #col3.pyplot(use_column_width=True)





#=======================================================================
def draw_squares(model, output, image, orig, min_intensity = 140,
                 min_size = 625, min_perc = 0.03,
                 cmap='inferno',
                 intensity_based_size = False, accuracy = 0,):
    """Function to quickly draw squares around area of interest in radio-image
    uses HeatMapSquares, GroupsM and GroupMatrix classes as well as the heatmap
    for finding intensity of pixels
    model - prediction model
    output - predicted class of image
    image - resized image used for prediction
    orig - original unchanged image
    min_intensity - minimum intensity required to become part of group [0-255]
    min_size = minimum size of group needed to have a square be drawn
    min_perc = minimum contribution requirement of a group to total intensity of an image
    cmap - matplotlib color map
    """

    # Find groups
    cam = GradCAM(model, output)
    heatmap = cam.compute_heatmap(image)
    group_class = Groupmatrix(heatmap, min_intensity)
    group_class.find_groups()
    out_groups = group_class.return_groups(min_size,min_perc)

    # Draw groups
    hs = HeatmapSquares(orig, out_groups)
    output_image = hs.draw_all_groups(intensity_based_size, accuracy,cmap)

    return output_image




def main():
    uploaded_image = upload_image()
    container = sider_bar()

    if st.sidebar.button('Predict'):
        show_prediction(uploaded_image, model="VGG16")

if __name__ == "__main__":
    main()