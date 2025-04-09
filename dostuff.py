from ResNet import ResNet18
from torch import nn

model=ResNet18(num_classes=10)
all_convs = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
print(all_convs)

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        print(name,layer.weight)
    