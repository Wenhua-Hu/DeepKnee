# -*- encoding: utf-8 -*-
from flask import render_template, request, jsonify
from jinja2 import TemplateNotFound

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from apps.home import blueprint
from apps.home.forms import SearchForm
from apps.home.models import query_patient, query_images, get_all_patients
from apps.home.torch_gradcam import img_preprocess, farward_hook, backward_hook, fmap_block, grad_block, cam_show_img
from apps.home.torch_models import get_model
from run import app


@blueprint.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    patient = query_patient("Tom Huang")
    images = query_images(patient.id)
    if form.validate_on_submit():
        name = form.search_name.data
        patient = query_patient(name)
        images = query_images(patient.id)
    patients = get_all_patients()

    return render_template('home/index.html', segment='index', title="DeepKnee", patient=patient, images=images,
                           form=form, patients=patients)


@blueprint.route('/<template>')
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:
        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


def allowed_file(filename):
    ALLOWED_EXTENSIONS = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(model_name="resnet34", filename='9001400L.png'):
    classes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    classes = list(classes.get(key) for key in range(5))
    IMAGES_PATH = app.config['IMAGES_PATH']

    knee_dir = os.path.join(IMAGES_PATH, 'knee')
    gradcam_dir = os.path.join(IMAGES_PATH, 'gradcam')

    knee_image = os.path.join(knee_dir, filename)

    gradcam_1 = os.path.splitext(filename)[0] + '_' + 'gradcam_1.png'
    gradcam_1_file_path = os.path.join(gradcam_dir, gradcam_1)

    img = cv2.imread(knee_image, 1)
    img_input = img_preprocess(img)

    net = get_model(model_name)

    if model_name == "resnet34":
        net.resnet_model.layer4[-1].register_forward_hook(farward_hook)
        net.resnet_model.layer4[-1].register_backward_hook(backward_hook)
        outputs = net(img_input)
        output = outputs[1]

    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    cam_show_img(img, fmap, grads_val, gradcam_1_file_path)

    y_proba = F.softmax(output, dim=-1)

    return y_proba, gradcam_1

@blueprint.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        filename = request.form.get("filename")

        if filename is None or filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(filename):
            return jsonify({'error': 'format not supported'})

        y_proba, gradcam_1 = model_predict("resnet34", filename)
        y = torch.squeeze(y_proba)
        label = torch.argmax(y)


        data = {
            'prediction': y.tolist(),
            'predicted_label': label.item(),
            'heatmap_1': gradcam_1
        }
        return jsonify(data)



