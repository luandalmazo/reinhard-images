import argparse
import torch
from train import train_model
from test import test_model
from transformed_dataset import TransformedDataset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

''' Calculates mean and standard deviation from a dataset '''
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

''' Manually fixing pytorch seed '''
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="A script to fetch, transform data, and train a model on it!")

''' Choices '''
parser.add_argument("--mode", choices=["normal", "reinhard"], default="normal", help="Choice between running with the Reinhard Normalization or not.")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs, default at 10")

''' Parse the arguments '''
args = parser.parse_args()
data_path = './Treinamento_4cls/'
mark = 'n'

if args.mode == "reinhard":
    #Path to reinhard-normalized data
    data_path = './norm2/'
    mark = 'r'

''' Declaring data transformations '''
transform = {
    'train_n': transforms.Compose([
        transforms.Resize((224, 224))
        ,transforms.ToTensor(),
        transforms.Normalize(mean=[0.6916, 0.6315, 0.6010], std=[0.1349, 0.1432, 0.1480])
    ])
    , 'train_r': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    ,'test_n': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.7027, 0.6437, 0.6121], std=[0.1264, 0.1345, 0.1400])
    ])
    ,'test_r': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    ,'neutral': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
}

''' Managing data '''
dataset = datasets.ImageFolder(data_path)

''' Splitting between training and testing data '''
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

dtrain, dtest = random_split(dataset, [train_size, test_size])

''' Applying data transformation '''
transformed_train_data = TransformedDataset(dtrain, transform=transform[f'train_{mark}'])
dataloader_train = DataLoader(transformed_train_data, batch_size=32, shuffle=True)
#meant, stdt = calculates_std_mean(dataloader_train)

''' Training the model '''
train_model(dataloader_train, args.num_epochs, args.mode)

''' Applying data transformation on testing data '''
transformed_test_data = TransformedDataset(dtest, transform=transform[f'test_{mark}'])
dataloader_test = DataLoader(transformed_test_data, batch_size=32, shuffle=True)

''' Testing the previously saved model on the given data '''
test_model(dataloader_test, dataset.classes, args.mode)
#meantest, stdtest = calculates_std_mean(dataloader_test)





