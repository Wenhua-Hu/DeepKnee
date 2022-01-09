import torch,os
import torch.nn as nn
from torchvision import models
from run import app

MODELS_PATH = app.config['MODELS_PATH']

class multi_output_model(nn.Module):
    def __init__(self, model_core, num_ftrs):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.y1o = nn.Linear(num_ftrs, 2)
        self.y2o = nn.Linear(num_ftrs, 5)

    def forward(self, x):
        x1 = self.resnet_model(x)

        y1o = self.y1o(x1)
        y2o = self.y2o(x1)

        return y1o, y2o


def get_model(model_name='resnet34'):
    if model_name.lower() == "resnet34":
        model = models.resnet34(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential()
        model = multi_output_model(model, num_features)
        model_name = "resnet34_0.pth"
        model_location = os.path.join(MODELS_PATH, model_name)
        model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))

    return model
