import torch
import torch.nn as nn
import torchvision
from torchvision import models
from .base import ModelSplitor

MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


AVAILABLE_POS = {
    "resnet18": [4, 5, 6, 7, 8, 9]
}


class ResNetModelSplitor(ModelSplitor):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.split_sign = torch.nn.Sequential

    def preprocess(self):
        self._modify_fc(self.num_classes)
        
    def _modify_fc(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                    nn.Flatten())

    def _get_available_split(self):
        statistic = []
        for idx, layer in enumerate(self.model.children()):
            if type(layer) == self.split_sign:
                statistic.append(idx)
        statistic.append(statistic[-1]+1)
        return statistic
    
    def get_available_split(self):
        return AVAILABLE_POS[self.model_name]
    
    def split_model(self, split_pos):
        self.model = MODELS[self.model_name](pretrained=True)
        self.preprocess()
        statistic = self.get_available_split()
        if split_pos > len(statistic):
            raise ValueError(f"split_pos should be less than {len(statistic)}")
        split_pos = statistic[split_pos]
        bottom_model = nn.Sequential(*list(self.model.children())[:split_pos])
        top_model = nn.Sequential(*list(self.model.children())[split_pos:])
        return bottom_model, top_model

