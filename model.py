from collections import OrderedDict
from torch import nn
from torchvision import models
import torch
from os import path

class Model():
    
    model = None

    def __init__(self):
        pass

    def get_model(self,hidden_units,file_name=""):
        if file_name == "":
            self.model = models.vgg11(pretrained=True)
            self.__disable_grads()
            self.model.classifier = self.__get_classifier(hidden_units)
            torch.save(self.model,'vgg11.pth')
            
        return self.model

    def __disable_grads(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def __get_classifier(self,hidden_units):
        return nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))