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
