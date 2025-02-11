from dotenv import load_dotenv
import torch
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torch import optim
import os
import matplotlib.pyplot as plt
from datetime import datetime
import math

VERSION='normal'

torch.manual_seed(42)
start_time = datetime.now()
print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

''' Check if GPU is available '''
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

''' Transform the images '''
transform = {
    'train_n': transforms.Compose([
        transforms.Resize((224, 224))
        ,transforms.ToTensor(),
        transforms.Normalize(mean=[0.6916, 0.6315, 0.6010], std=[0.1349, 0.1432, 0.1480])
    ])
    , 'train_r': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.6928, 0.6282, 0.5761], std=[0.2531, 0.2731, 0.2738])
    ])
}

''' Train a new model on the given data '''
def train_model(dataloader_train, epochs, mode): 
    ''' Calculate the mean and standard deviation'''
    #mean, std = calculates_std_mean(dataloader_train) # mean -> [0.6972, 0.6460, 0.6218] std -> [0.1117, 0.1153, 0.1183]
    ''' Load the model '''
    device = get_device()
    print("Running Train on", device)

    ''' Set the model '''
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048, 4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    ''' Freezes the weights of the model '''
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    model.to(device)

    train_loss = []
    test_loss = []
    best_loss = math.inf
    train_acc = []
    best_acc = 0.0

    if mode == 'normal':
        mark = 'N'
    else:
        mark = 'R'


    ''' Train the model '''
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        running_acc = 0.0

        with torch.set_grad_enabled(True):
            for images, labels in dataloader_train:

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                
                running_acc += torch.sum(preds == labels.data)
            ''' Calculate the loss '''
            epoch_loss = running_loss / len(dataloader_train.dataset)
            train_loss.append(epoch_loss)

            ''' Calculate the training acc '''
            epoch_acc = running_acc.double() / len(dataloader_train.dataset)
            train_acc.append(epoch_acc.cpu().numpy())
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            ''' Print the loss '''
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss}, Accuracy: {epoch_acc:.4f}")

            ''' Save the best model '''
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'model{mark}.pth')

            print("Time elapsed: ", datetime.now() - start_time)
            print("----------------------------------------------")

    ''' Plot loss '''
    plt.figure()
    plt.plot(train_loss, label='Training loss') 
    plt.legend()
    plt.savefig(f'loss{mark}.png')
    plt.close()

    ''' Plot accuracy '''
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.legend()
    plt.savefig(f'train_acc{mark}.png')
    plt.close()

    End = datetime.now()
    print(f"End: {End.strftime('%Y-%m-%d %H:%M:%S')}")
