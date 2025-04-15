import pickle
from PrunUtillCP import channel_prune, channel_prune_resnet18, get_num_channels_to_keep
from ResNet import ResNet18
from torch import nn

from Utill import print_model
from VGG import VGG

model=ResNet18(num_classes=10)
# all_convs = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
# print(all_convs)

# for name, layer in model.named_modules():
#     if isinstance(layer, nn.Conv2d):
#         print(name,layer.weight)
    
    
# for name, param in model.named_modules():
#     if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
#          print(f'\'{name}\':0.90,')
    
# accuracies_path='checkpoint/Resnet-18/Resnet-18_accuracies.pkl'
# with open(accuracies_path, "rb") as f:
#     sparsities = pickle.load(f)
# for s in sparsities:
#     s=[ (i,a) for i,a in enumerate(s) if a>94.79]
#     print(s[-1][0]+1)
sparsity_dict = {      #for F
'conv1':0.85,
'layer1.0.conv1':0.90,
'layer1.0.conv2':0.90,
'layer1.1.conv1':0.90,
'layer1.1.conv2':0.90,
'layer2.0.conv1':0.75,
'layer2.0.conv2':0.90,
'layer2.1.conv1':0.80,
'layer2.1.conv2':0.70,
'layer3.0.conv1':0.65,
'layer3.0.conv2':0.90,
'layer3.1.conv1':0.80,
'layer3.1.conv2':0.80,
'layer4.0.conv1':0.90,
'layer4.0.conv2':0.95,
'layer4.1.conv1':0.95,
'layer4.1.conv2':0.95,
}

print_model(model)
            

model=channel_prune(model,sparsity_dict,'Resnet-18')

print_model(model)

