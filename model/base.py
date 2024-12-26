import torch
import torch.nn as nn
import torchvision
from torchvision import models
from abc import abstractmethod

class ModelSplitor():

    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

    def preprocess(self):
        pass
        
    def _get_available_split(self):
        pass

    def get_available_split(self):
        raise NotImplementedError("get_availabel_split method not implemented")
    
    def split_model(self, split_pos):
        raise NotImplementedError("split_model method not implemented.")

