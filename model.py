from collections import OrderedDict
from torch import nn
from torchvision import models
import torch
from os import path

class Model():
    
    model = None

    def __init__(self):
        pass

    def get_model(self,hidden_units,arch_name="vgg11"):
        if arch_name == "vgg11":
            self.model = models.vgg11(pretrained=True)
        elif arch_name == "vgg13":
            self.model = models.vgg13(pretrained=True)

        self.__disable_grads()
        self.model.__get_classifier(hidden_units)
        torch.save(self.model, arch_name + '.pth')
            
        return self.model

    def __disable_grads(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def __get_classifier(self,hidden_units):
        input_size = model.classifier[0].in_features
        return nn.Sequential(nn.Linear(input_size, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))