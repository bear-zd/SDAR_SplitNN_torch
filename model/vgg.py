import torch
import torch.nn as nn
import torchvision
from torchvision import models
from .base import ModelSplitor
MODELS = {
    "vgg19": models.vgg19,
    "vgg16": models.vgg16,
}

AVAILABLE_POS = {
    "vgg16": [4, 9, 16, 23, 30, 31],
    "vgg19": [4, 9, 18, 27, 36, 37]
}



class VGGModelSplitor(ModelSplitor):
    def __init__(self, model_name, num_classes):
        super().__init__(model_name, num_classes)
        self.split_sign = torch.nn.MaxPool2d

    def preprocess(self):
        self._modify_fc(self.num_classes)
        self._preproccess_vgg()

    def _preproccess_vgg(self):
        model_list = []
        for layers in list(self.model.children()):
            if type(layers) == torch.nn.Sequential:
                for layer in layers:
                    model_list.append(layer)
            else:
                model_list.append(layers)
        self.model = nn.Sequential(*model_list)
        
    def _modify_fc(self, num_classes):
        if "resnet" in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Flatten())
        else:
            self.model.avgpool = nn.Sequential(self.model.avgpool, nn.Flatten())
            self.model.classifier.append(nn.Linear(1000, num_classes)) 

    def _get_available_split(self):
        temp_model = MODELS[self.model_name](pretrained=True)
        statistic = []
        for idx, layer in enumerate(list(temp_model.children())[0]):
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
