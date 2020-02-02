import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import pandas as pd
import numpy as np
from PIL import Image
import json
from model import Model
import argparse
import json
from model import Model


def load_checkpoint(filepath,arch):
    checkpoint = torch.load(filepath)
    
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)    
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)    
        
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    
    for param in model.parameters(): 
        param.requires_grad = False #turning off tuning of the model
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''      
    pil_image = Image.open(image)
    size = 256,256
    
    pil_image.thumbnail(size, Image.ANTIALIAS)

    width, height = pil_image.size   # Get dimensions

    # Center crop dimentions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    pil_image = pil_image.crop((left, top, right, bottom))
    
    np_image = np.array(pil_image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Normalisation
    np_image = (np_image - mean)/std
    
    # Transpose
    np_image = np_image.transpose ((2,0,1))
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    im = torch.from_numpy(image).type(torch.FloatTensor)
    im = im.unsqueeze (dim = 0)
    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp(output)
    probs, indeces = output_prob.topk (topk)
    
    probs = probs.numpy () #converting both to numpy array
    indeces = indeces.numpy () 
    
    probs = probs.tolist () [0] #converting both to list
    indeces = indeces.tolist () [0]
    
    mapping = {val: key for key, val in
            model.class_to_idx.items()
            }
    
    classes = [mapping [item] for item in indeces]
    classes = np.array (classes) #converting to Numpy array
    return probs, classes
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,help="Image file",required=True)
    parser.add_argument("-c", "--checkpoint", type=str,help="Path to model file",required=True)
    parser.add_argument("-tk", "--top_k", type=int,help="Number of results",required=False,default=3)
    parser.add_argument("-cat", "--category_names", type=str,help="Category name json file",required=True)
    parser.add_argument("-gpu", "--gpu", choices=['cpu','cuda'],default='cpu', required=False) 
    parser.add_argument("-arch", "--arch", choices=['vgg11','vgg13'], required=True)

    global_args = parser.parse_args()
    
    model = load_checkpoint(global_args.checkpoint)
        
    probs, classes = predict(global_args.input, model, topk=global_args.top_k)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = [cat_to_name [item] for item in classes]
    print(class_names)
    
