import pickle

import torch
from Model_Evaluation import get_model_macs, get_model_size, get_model_sparsity
from PrunUtillCP import channel_prune 
from ResNet import *
from torch import nn

from Utill import print_model
from VGG import VGG
from Viewer import accumulate_plot_figures

model=ResNet34(classes=10)
# all_convs = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
# print(all_convs)

for name, layer in model.named_parameters():
    print(f"{name}")

    
for name, layer in model.named_children():
    if isinstance(layer,nn.Sequential):
        for sub_name, sub_layer in layer.named_children():
            print(f'\'{name}.{sub_name}\':0.90,')

for name, param in model.named_modules():
    if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
         print(f'\'{name}\':0.90,')
    
# accuracies_path='checkpoint/Resnet-18/Resnet-18_accuracies.pkl'
# with open(accuracies_path, "rb") as f:
#     sparsities = pickle.load(f)
# for s in sparsities:
#     s=[ (i,a) for i,a in enumerate(s) if a>94.79]
#     print(s[-1][0]+1)


sparsity_dict = {      #for F
'conv1':0.85,
'layer1.0.conv1':0.90,
'layer1.1.conv1':0.90,
'layer2.0.conv1':0.75,
'layer2.1.conv1':0.80,
'layer3.0.conv1':0.65,
'layer3.1.conv1':0.80,
'layer4.0.conv1':0.90,
'layer4.1.conv1':0.95,
}

model.to('cuda')
print_model(model)
model.eval()
# input_tensor=torch.randn(1, 3, 32, 32).to('cuda')
# output = model(input_tensor)  # Ensure input_tensor is properly formatted

# print(get_model_macs(model))
# sparsity =get_model_sparsity(model)
# model_size =get_model_size(model,count_nonzero_only=True)
# print(sparsity)
# model=channel_prune(model,0.7,'Resnet-34')
# print(model_size)
# print_model(model)

# input_tensor=torch.randn(1, 3, 32, 32).to('cuda')
# output = model(input_tensor)  # Ensure input_tensor is properly formatted
# print(output)
i=0
for name, layer in model.named_children():
    if isinstance(layer,nn.Sequential):
        for l in layer:
            print(i,l)
            i=i+1

accumulate_plot_figures('checkpoint/Resnet-34/CP/sensitivity_curves')
