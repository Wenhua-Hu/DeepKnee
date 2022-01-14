# -*- encoding: utf-8 -*-

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import current_app
from flask import render_template, request, jsonify

from apps.home import blueprint
from apps.home.forms import SearchForm
from apps.home.models import query_patient, query_images, get_all_patients
from apps.torch_utils import lime_
from apps.torch_utils.gradcam import gradcam_resnet
from apps.torch_utils.models_ import load_model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


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


@blueprint.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    patient = query_patient("Tom Huang")

    images = query_images(patient.id)
    if form.validate_on_submit():
        name = form.search_name.data
        patient = query_patient(name)
        if patient is None:
            patient = query_patient("Tom Huang")
        images = query_images(patient.id)
    patients = get_all_patients()

    return render_template('home/index.html', segment='index', title="DeepKnee", patient=patient, images=images,
                           form=form, patients=patients)


@blueprint.route('/predict_lime', methods=['POST', 'GET'])
def predict_lime():
    if request.method != 'POST':
        return jsonify({'error': 'request method is not proper'})

    filename = request.form.get("filename")
    modelname = request.form.get("modelname")

    IMAGES_KNEE_ORIGINAL = current_app.config['IMAGES_KNEE_ORIGINAL']
    IMAGES_KNEE_LIME = current_app.config['IMAGES_KNEE_LIME']

    img_path = os.path.join(IMAGES_KNEE_ORIGINAL, filename)

    output_lime = os.path.join(IMAGES_KNEE_LIME, os.path.splitext(filename)[0] + '_lime.png')

    model = load_model(modelname, current_app.config[modelname.upper()])

    clime = lime_.CalLime(model, img_path, output_lime)
    clime.lime_predict()

    data = {
        'output_lime': output_lime
    }
    return jsonify(data)


@blueprint.route('/predict_score', methods=['POST', 'GET'])
def predict_score():
    if request.method != 'POST':
        return jsonify({'error': 'request method is not proper'})

    filename = request.form.get("filename")
    modelname = request.form.get("modelname")

    IMAGES_KNEE_ORIGINAL = current_app.config['IMAGES_KNEE_ORIGINAL']

    img_path = os.path.join(IMAGES_KNEE_ORIGINAL, filename)

    img = cv2.imread(img_path, 1)
    img_input = img_preprocess(img)

    model = load_model(modelname, current_app.config[modelname.upper()])

    model.eval()
    model.zero_grad()

    if model.name == "resnet34":
        _, output = model(img_input)

    y_proba = F.softmax(output, dim=-1)
    y = torch.squeeze(y_proba)
    label = torch.argmax(y)

    data = {
        'prediction': y.tolist(),
        'predicted_label': label.item()
    }
    return jsonify(data)


@blueprint.route('/predict_gradcam', methods=['POST', 'GET'])
def predict_gradcam():
    if request.method != 'POST':
        return jsonify({'error': 'request method is not proper'})

    filename = request.form.get("filename")
    modelname = request.form.get("modelname")

    IMAGES_KNEE_ORIGINAL = current_app.config['IMAGES_KNEE_ORIGINAL']
    IMAGES_KNEE_GRADCAM = current_app.config['IMAGES_KNEE_GRADCAM']

    img_path = os.path.join(IMAGES_KNEE_ORIGINAL, filename)
    output1 = os.path.join(IMAGES_KNEE_GRADCAM, os.path.splitext(filename)[0] + '_gradcam_1.png')
    output2 = os.path.join(IMAGES_KNEE_GRADCAM, os.path.splitext(filename)[0] + '_gradcam_2.png')
    output3 = os.path.join(IMAGES_KNEE_GRADCAM, os.path.splitext(filename)[0] + '_gradcam_3.png')

    model = load_model(modelname, current_app.config[modelname.upper()])

    if modelname == 'resnet34' or model.name == 'resnet34':
        gradcam_model = gradcam_resnet(model, filename, output1, output2, output3)

    gradcam_model(img_path)

    data = {
        'output1': output1,
        'output2': output2,
        'output3': output3
    }

    return jsonify(data)
