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

global_args = None
model = None

def normalization_params():
    return [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]

def get_train_transforms():
    n1, n2 = normalization_params()
    return [transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(n1,n2)]

def get_test_transforms():
    n1, n2 = normalization_params()
    return [transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
     transforms.Normalize(n1,n2)]

def get_transformed_data(transform_list,data_dir):
    data_transforms = transforms.Compose(transform_list)
    transformed_data = datasets.ImageFolder(data_dir, transform=data_transforms)
    return transformed_data

def get_data_loader(transform_list,data_dir,shuffle=False):
    transformed_data = get_transformed_data(transform_list,data_dir)
    data_loader = torch.utils.data.DataLoader(transformed_data, batch_size=64, shuffle=shuffle)
    return data_loader

def prepare_training_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_loader = get_data_loader(get_train_transforms(),train_dir,shuffle=True)
    test_loader = get_data_loader(get_test_transforms(), test_dir)
    validation_loader = get_data_loader(get_test_transforms(), valid_dir)
 
    return train_loader, test_loader, validation_loader

def train_model():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    device = global_args.gpu        
    if global_args.gpu == 'cuda':
        device =  global_args.gpu if torch.cuda.is_available() else 'cpu'
            
    data_dir = global_args.data_dir 
    learn_rate = global_args.learning_rate 
    
    print(f'device -> {device}')
    print(f"data_dir -> {data_dir}")
    print(f"learn_rate -> {learn_rate}")    
    print("Moving model to " + device)
    model.to(device);
    trainloader, testloader, validationloader = prepare_training_data(data_dir)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=global_args.learning_rate)

    print("Starting Training Process")
    epochs = global_args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    print(f"epochs -> {epochs}")
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(validationloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(validationloader):.3f}")
                running_loss = 0
                model.train()


def test_training():
    device = global_args.gpu
    if global_args.gpu == 'cuda':
        device =  global_args.gpu if torch.cuda.is_available() else 'cpu'
    
    
    test_loss = 0
    accuracy = 0
    data_dir = global_args.data_dir

    criterion = nn.NLLLoss()
    trainloader, testloader, validationloader = prepare_training_data(data_dir)

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))
    
def persist_model():
    model.to ('cpu')
    train_dir = global_args.data_dir + '/train'    
    train_data = get_transformed_data(get_train_transforms(),train_dir)
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'mapping':    model.class_to_idx
             }
        
    optimizer = optim.Adam(model.classifier.parameters(), lr=global_args.learning_rate)
    torch.save (checkpoint, global_args.save_dir +'/checkpoint.pth')
    torch.save(optimizer.state_dict(), global_args.save_dir +'/optimizer.pth')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str,help="Enter Data Directory",required=True)
    parser.add_argument("-o", "--save_dir", type=str,help="Output Model Directory",required=True)
    parser.add_argument("-lr", "--learning_rate", type=float,help="Learning rate",required=False,default=0.003)
    parser.add_argument("-hu", "--hidden_units", type=int,help="Number of Hidden units",required=False,default=158)
    parser.add_argument("-e", "--epochs", type=int,help="Number of epochs",required=False,default=1)
    parser.add_argument("-gpu", "--gpu", choices=['cpu','cuda'],default='cpu', required=False) 
    parser.add_argument("-arch", "--arch", choices=['vgg11','vgg13'], required=True)
    
    global_args = parser.parse_args()
    
    model_loader = Model()
    model = model_loader.get_model(global_args.hidden_units,arch_name=global_args.arch)
    
    train_model()
    test_training()
    persist_model()
