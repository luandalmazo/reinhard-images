from dotenv import load_dotenv
import torch
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch import optim
import os
import matplotlib.pyplot as plt
from datetime import datetime
import math

start_time = datetime.now()
print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

''' Calculate the mean and standard deviation of the dataset '''
def calculates_std_mean(dataloader):
    mean = 0.
    std = 0.
    num_samples = 0.

    for data, _ in dataloader:

        batch_samples = data.size(0)

        ''' Redimension the data to (number of samples, number of channels, number of pixels) '''
        data = data.view(batch_samples, data.size(1), -1)

        ''' Update the mean and standard deviation '''
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    return mean, std

''' Check if GPU is available '''
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
''' Load the environment variables '''
load_dotenv()
TRAIN_DIR = os.getenv('TRAIN_DIR')
TEST_DIR = os.getenv('TEST_DIR')
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))

''' Transform the images '''
transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.6972, 0.6460, 0.6218], std=[0.1117, 0.1153, 0.1183])
    ])
}

''' Load the dataset '''
dataset_train = datasets.ImageFolder(TRAIN_DIR, transform=transform['train'])

''' Create the dataloader '''
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)   

''' Calculate the mean and standard deviation'''
mean, std = calculates_std_mean(dataloader_train) # mean -> [0.6972, 0.6460, 0.6218] std -> [0.1117, 0.1153, 0.1183]

''' Load the model '''
device = get_device()
print("Running on", device)

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

''' Train the model '''
for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0.0

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

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

        ''' Calculate the loss '''
        epoch_loss = running_loss / len(dataloader_train.dataset)
        train_loss.append(epoch_loss)

        ''' Print the loss '''
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss}")

        ''' Save the best model '''
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'model.pth')

        print("Time elapsed: ", datetime.now() - start_time)
        print("----------------------------------------------")

''' Plot the loss '''
plt.plot(train_loss, label='Training loss') 
plt.legend()
plt.savefig('loss.png')

End = datetime.now()
print(f"End: {End.strftime('%Y-%m-%d %H:%M:%S')}")