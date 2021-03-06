# -*- encoding: utf-8 -*-


import os
from decouple import config

class Config(object):

    basedir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(basedir, 'data')

    # Set up the App SECRET_KEY
    SECRET_KEY = config('SECRET_KEY', default='S#perS3crEt_007')

    # This will create a file in <app> FOLDER
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(datadir, 'db/app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # the type of images
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # directory list
    MODELS_PATH = os.path.join(datadir, 'models')

    imagedir = os.path.join(basedir, 'static/assets/images')
    IMAGES_KNEE_ORIGINAL = os.path.join(imagedir, 'knee_original')
    IMAGES_KNEE_LIME = os.path.join(imagedir, 'knee_lime')
    IMAGES_KNEE_GRADCAM = os.path.join(imagedir, 'knee_gradcam')
    IMAGES_KNEE_BBOX = os.path.join(imagedir, 'knee_boundingbox')

    # add models
    VGG16 = os.path.join(MODELS_PATH, 'vgg16_1.pth')
    VGG19 = os.path.join(MODELS_PATH, 'vgg19_1.pth')
    RESNET18 = os.path.join(MODELS_PATH, 'resnet18_0.pth')
    RESNET34 = os.path.join(MODELS_PATH, 'resnet34_1.pth')
    RESNET50 = os.path.join(MODELS_PATH, 'resnet50_1.pth')
    RESNET101 = os.path.join(MODELS_PATH, 'resnet101_1.pth')


class ProductionConfig(Config):
    DEBUG = False

    # Security
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = 0

class DebugConfig(Config):
    DEBUG = True


# Load all possible configurations
config_dict = {
    'Production': ProductionConfig,
    'Debug': DebugConfig
}
