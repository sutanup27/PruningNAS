from ResNet import ResNet18
from torch import nn

from VGG import VGG

model=VGG(classes=10)
# all_convs = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
# print(all_convs)

# for name, layer in model.named_modules():
#     if isinstance(layer, nn.Conv2d):
#         print(name,layer.weight)
    
    
# for name, param in model.named_modules():
#     if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear): # we only prune conv and fc weights
#          print(f'\'{name}\':0.90,')
    
    
accuracy=[.2,.3,.5]
print(type(accuracy))