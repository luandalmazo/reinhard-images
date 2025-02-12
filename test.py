import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, f1_score, recall_score
from torch.utils.data import DataLoader
from transformed_dataset import TransformedDataset
import matplotlib.pyplot as plt
import seaborn as sns

''' Defining some plotting functions '''
def plot_confusion_matrix(y_true, y_pred, class_names, label):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix {label}")
    plt.savefig(f"confusion_matrix_{label}.png")
    plt.close()

def plot_roc_curve(y_true, y_probs, class_names, label):
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true = np.array(y_true)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {label}")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{label}.png")
    plt.close()
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
    ,'test_r': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.6921, 0.6275, 0.5766], std=[0.2520, 0.2715, 0.2727])
    ])
    ,'neutral': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

}

''' Set the network to test all the given models '''
def test_model(dataloader_test, classes, mode):
    device = get_device()
    if mode == 'normal':
        mark = 'N'
    else:
        mark = 'R'
    model_paths = [f'model{mark}.pth']

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