# DDDwithPruningArchitectures

Parameter configuration for Resnet18:
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size=64
num_epochs=200

optimizer = SGD( model.parameters(), lr=0.1,  momentum=0.9,  weight_decay=5e-4,)
scheduler = CosineAnnealingLR(optimizer, num_epochs)
