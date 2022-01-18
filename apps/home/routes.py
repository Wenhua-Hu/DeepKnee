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
from apps.torch_utils.bounding_box import draw_boundingbox
from apps.torch_utils.gradcam import get_gradcam
from apps.torch_utils.lime_ import lime_run
from apps.torch_utils.models_ import load_model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


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


def model_predict(model, img_input):
    model.eval()
    model.zero_grad()

    # add models
    if model.name in ("resnet18", 'resnet50'):
        output = model(img_input)
    else:
        output = model(img_input)

    y_proba = F.softmax(output, dim=-1)
    y = torch.squeeze(y_proba)
    label = torch.argmax(y)

    return y, label


def get_request_data(request):
    if request.method != 'POST':
        return jsonify({'error': 'request method is not proper'})

    filename = request.form.get("filename")
    modelname = request.form.get("modelname")
    if 'nsamples' in list(request.form.keys()):
        nsamples = int(request.form.get("nsamples"))
    else:
        nsamples = 0


    return filename, modelname, nsamples


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


    IMAGES_KNEE_ORIGINAL = current_app.config['IMAGES_KNEE_ORIGINAL']
    IMAGES_KNEE_LIME = current_app.config['IMAGES_KNEE_LIME']

    filename, modelname, num_samples = get_request_data(request)



    img_path = os.path.join(IMAGES_KNEE_ORIGINAL, filename)

    model = load_model(modelname, current_app.config[modelname.upper()])

    lime_run(model, img_path, IMAGES_KNEE_LIME, num_samples)

    data = {
        'output_lime': os.path.splitext(filename)[0] + '_lime.png'
    }
    return jsonify(data)


@blueprint.route('/predict_score', methods=['POST', 'GET'])
def predict_score():
    IMAGES_KNEE_ORIGINAL = current_app.config['IMAGES_KNEE_ORIGINAL']

    filename, modelname, _ = get_request_data(request)

    img_path = os.path.join(IMAGES_KNEE_ORIGINAL, filename)
    img_input = img_preprocess(img_path)

    model = load_model(modelname, current_app.config[modelname.upper()])
    y, label = model_predict(model, img_input)

    return jsonify({'prediction': y.tolist(), 'predicted_label': label.item()})


@blueprint.route('/predict_gradcam', methods=['POST', 'GET'])
def predict_gradcam():
    IMAGES_KNEE_ORIGINAL = current_app.config['IMAGES_KNEE_ORIGINAL']
    IMAGES_KNEE_GRADCAM = current_app.config['IMAGES_KNEE_GRADCAM']
    IMAGES_KNEE_BBOX = current_app.config['IMAGES_KNEE_BBOX']

    filename, modelname, _ = get_request_data(request)

    img_path = os.path.join(IMAGES_KNEE_ORIGINAL, filename)
    model = load_model(modelname, current_app.config[modelname.upper()])

    cam_ = get_gradcam(model, img_path, IMAGES_KNEE_GRADCAM)
    draw_boundingbox(cam_, img_path, IMAGES_KNEE_BBOX)

    data = dict()
    data['output_gradcam'] = os.path.splitext(filename)[0] + '_gradcam.png'
    data['output_bbox'] = os.path.splitext(filename)[0] + '_boundingbox.png'
    return jsonify(data)
