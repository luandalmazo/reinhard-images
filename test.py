import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, f1_score, recall_score
from torch.utils.data import DataLoader

''' Check if GPU is available '''
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
''' Test the model '''
def test_model(model, dataloader, device):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            y_true.append(target.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    precision_class = precision_score(y_true, y_pred, average=None)
    recall_class = recall_score(y_true, y_pred, average=None)
    f1_class = f1_score(y_true, y_pred, average=None)

    return precision, recall, f1, precision_class, recall_class, f1_class


''' Transform the images '''
set_transform = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.6972, 0.6460, 0.6218], std=[0.1117, 0.1153, 0.1183])
    ])
}

''' Load the environment variables '''
load_dotenv()
TEST_DIR = os.getenv('TEST_DIR')

''' Load the dataset '''
dataset_test = datasets.ImageFolder(TEST_DIR, transform=set_transform['test'])
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

device = get_device()
model_paths = ['model_1.pth', 'model_2.pth', 'model_3.pth, model_4.pth']

all_precisions = []
all_recalls = []
all_f1s = []
all_precisions_class = []
all_recalls_class = []
all_f1s_class = []

for model_path in model_paths:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 4)
    model.load_state_dict(torch.load(model_path))

    precision, recall, f1, precision_class, recall_class, f1_class = test_model(model, dataloader_test, device)

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)
    all_precisions_class.append(precision_class)
    all_recalls_class.append(recall_class)
    all_f1s_class.append(f1_class)

mean_precision = np.mean(all_precisions)
std_precision = np.std(all_precisions)

mean_recall = np.mean(all_recalls)
std_recall = np.std(all_recalls)

mean_f1 = np.mean(all_f1s)
std_f1 = np.std(all_f1s)

mean_precision_class = np.mean(all_precisions_class, axis=0)
std_precision_class = np.std(all_precisions_class, axis=0)

mean_recall_class = np.mean(all_recalls_class, axis=0)
std_recall_class = np.std(all_recalls_class, axis=0)

mean_f1_class = np.mean(all_f1s_class, axis=0)
std_f1_class = np.std(all_f1s_class, axis=0)

print("--------------------------------------")
print(f"Precision: {mean_precision:.2f} ± {std_precision:.2f}")
print(f"Recall: {mean_recall:.2f} ± {std_recall:.2f}")
print(f"F1: {mean_f1:.2f} ± {std_f1:.2f}")

print("Precision per class")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {mean_precision_class[idx]:.2f} ± {std_precision_class[idx]:.2f}")

print("Recall per class")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {mean_recall_class[idx]:.2f} ± {std_recall_class[idx]:.2f}")

print("F1 per class")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {mean_f1_class[idx]:.2f} ± {std_f1_class[idx]:.2f}")