# -*- encoding: utf-8 -*-


import os
from decouple import config

class Config(object):

    basedir = os.path.abspath(os.path.dirname(__file__))

    # Set up the App SECRET_KEY
    SECRET_KEY = config('SECRET_KEY', default='S#perS3crEt_007')

    # This will create a file in <app> FOLDER
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # the type of images
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # directory list
    datadir = os.path.join(basedir, 'data')
    MODELS_PATH = os.path.join(datadir, 'models')
    IMAGES_KNEE_ORIGINAL = os.path.join(datadir, 'knee_original')
    IMAGES_KNEE_LIME = os.path.join(datadir, 'knee_lime')
    IMAGES_KNEE_GRADCAM = os.path.join(datadir, 'knee_gradcam')
    IMAGES_KNEE_BBOX = os.path.join(datadir, 'knee_boundingbox')

    RESNET34 = os.path.join(MODELS_PATH, 'resnet_0.pth')


class ProductionConfig(Config):
    DEBUG = False

    # Security
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = 0

    # PostgreSQL database
    # SQLALCHEMY_DATABASE_URI = '{}://{}:{}@{}:{}/{}'.format(
    #     config('DB_ENGINE', default='postgresql'),
    #     config('DB_USERNAME', default='whu'),
    #     config('DB_PASS', default='pass'),
    #     config('DB_HOST', default='localhost'),
    #     config('DB_PORT', default=5432),
    #     config('DB_NAME', default='deepknee-app')
    # )


class DebugConfig(Config):
    DEBUG = True


# Load all possible configurations
config_dict = {
    'Production': ProductionConfig,
    'Debug': DebugConfig
}
