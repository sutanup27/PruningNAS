import copy
import random


import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from DataPreprocessing import get_dataloaders
from ResNet import ResNet18
from TrainingModules import evaluate
from VGG import VGG
from TrainingModules import Training
from Viewer import plot_accuracy, plot_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

seed=0
random.seed(seed)

path='../dataset/cifar10'
classes=10
train_dataloader,test_dataloader=get_dataloaders(path,batch_size=64)

select_model='Resnet-18'
if select_model=='Vgg-16':
    model=VGG(classes=classes)
elif select_model=='Resnet-18':
    model = ResNet18()
    model.fc = torch.nn.Linear(model.fc.in_features, classes)  # num_classes is the number of output classes
else:
    print("Model does not exists")
    exit

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = SGD( model.parameters(), lr=0.1,  momentum=0.9,  weight_decay=5e-4,)

num_epochs=20
scheduler = CosineAnnealingLR(optimizer, num_epochs)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_model, losses, test_losses, accs=Training( model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs,scheduler=scheduler)

model=copy.deepcopy(best_model)
metric,_ = evaluate(model, test_dataloader)
print(f"Best model accuray:", metric)

plot_accuracy(accs)
plot_loss(losses,test_losses)

torch.save(model, f'./checkpoint/{select_model}/{select_model}_cifar_{metric}.pth')


    
