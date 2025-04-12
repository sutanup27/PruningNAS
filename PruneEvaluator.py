

import copy
from matplotlib import path
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from DataPreprocessing import get_dataloaders
from PrunUtill import ChannelPrunner, apply_channel_sorting
from ResNet import ResNet18
from TrainingModules import evaluate
from Utill import print_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sparsity_dict = {
'backbone.conv0.weight': 0.4,
'backbone.conv1.weight': 0.90,
'backbone.conv2.weight': 0.80,
'backbone.conv3.weight': 0.6,
'backbone.conv4.weight': 0.9,
'backbone.conv5.weight': 0.9,
'backbone.conv6.weight': 0.8,
'backbone.conv7.weight': 0.95,
'backbone.conv8.weight': 0.95,
'backbone.conv9.weight': 0.97,
'fc2.weight': 0.95,
}
path='../dataset/cifar10'
classes=10
select_model='resnet18'
model=ResNet18(num_classes=10)
model_path='checkpoint/resnet18/resnet18_cifar_87.93000030517578.pth'
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
model.to(device)
train_dataloader,test_dataloader=get_dataloaders(path) # Basemodel
# dense_model_accuracy,_=evaluate(model,test_dataloader)
# print('dense_model_accuracy:',dense_model_accuracy)
# pruned_model=copy.deepcopy(model)
# pruned_model = ChannelPrunner(pruned_model, 0.5,model_type=select_model)
# pruned_model_accuracy,_=evaluate(pruned_model,test_dataloader)
# print('sorted_model_accuracy:',pruned_model_accuracy)

print_model(model)
# pruned_model = ChannelPrunner(model, 0.3,model_type=select_model)
# print_model(pruned_model)

for name, layer in model.named_children():
    print(name, "->", type(layer))  
