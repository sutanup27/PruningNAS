# DDDwithPruningArchitectures
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size=64
## Parameter configuration for VGG-16:
num_epochs=20
optimizer = SGD( model.parameters(), lr=0.1,  momentum=0.9,  weight_decay=5e-4,)
scheduler = CosineAnnealingLR(optimizer, num_epochs)
FGP:
sparsity_dict = {       #for VGG
'backbone.conv0':0.80,
'backbone.conv1':0.90,
'backbone.conv2':0.80,
'backbone.conv3':0.75,
'backbone.conv4':0.70,
'backbone.conv5':0.80,
'backbone.conv6':0.80,
'backbone.conv7':0.95,
'fc2':0.90,
}

CP:
 sparsity_dict = {       #for VGG
 'backbone.conv0':0.70,
 'backbone.conv1':0.80,
 'backbone.conv2':0.80,
 'backbone.conv3':0.7,
 'backbone.conv4':0.70,
 'backbone.conv5':0.80,
 'backbone.conv6':0.80,
 'backbone.conv7':0.80,
 'fc2':0.8,
 }

## Parameter configuration for Resnet18:
num_epochs=200
optimizer = SGD( model.parameters(), lr=0.1,  momentum=0.9,  weight_decay=5e-4,)
scheduler = CosineAnnealingLR(optimizer, num_epochs)

FGP:
epoch=20
optimizer = SGD( model.parameters(), lr=0.001,  momentum=0.9,  weight_decay=5e-4,) 
<!-- optimizer = SGD( model.parameters(), lr=0.0001,  momentum=0.9,  weight_decay=5e-4,)  -->
scheduler = CosineAnnealingLR(optimizer, num_epochs)

sparsity_dict = {      #for F
'conv1':0.85,
'layer1.0.conv1':0.90,
'layer1.0.conv2':0.90,
'layer1.1.conv1':0.90,
'layer1.1.conv2':0.90,
'layer2.0.conv1':0.75,
'layer2.0.conv2':0.90,
'layer2.0.shortcut.0':0.80,
'layer2.1.conv1':0.80,
'layer2.1.conv2':0.70,
'layer3.0.conv1':0.65,
'layer3.0.conv2':0.90,
'layer3.0.shortcut.0':0.75,
'layer3.1.conv1':0.80,
'layer3.1.conv2':0.80,
'layer4.0.conv1':0.90,
'layer4.0.conv2':0.95,
'layer4.0.shortcut.0':0.95,
'layer4.1.conv1':0.95,
'layer4.1.conv2':0.95,
'fc':0.80,
}
