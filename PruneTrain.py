from torch import nn
import copy
import random
import torch
from DataPreprocessing import get_dataloaders
from Model_Evaluation import get_model_size, get_sparsity
from PrunUtillCP import ChannelPrunner
from PrunUtillFGP import FineGrainedPruner
from TrainingModules import TrainingPrunned, evaluate
from Utill import print_model
from Viewer import plot_accuracy, plot_loss  # Ensure you import your correct model architecture
seed=0
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
# Initialize the model
path='../dataset/cifar10'
select_model='Vgg-16'
pruning_type='CP'
#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='checkpoint/Vgg-16/Vgg-16_cifar_93.0199966430664.pth'
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary

model.to(device)



# sparsity_dict = {      #for F
# 'conv1':0.85,
# 'layer1.0.conv1':0.90,
# 'layer1.0.conv2':0.90,
# 'layer1.1.conv1':0.90,
# 'layer1.1.conv2':0.90,
# 'layer2.0.conv1':0.75,
# 'layer2.0.conv2':0.90,
# 'layer2.0.shortcut.0':0.80,
# 'layer2.1.conv1':0.80,
# 'layer2.1.conv2':0.70,
# 'layer3.0.conv1':0.65,
# 'layer3.0.conv2':0.90,
# 'layer3.0.shortcut.0':0.75,
# 'layer3.1.conv1':0.80,
# 'layer3.1.conv2':0.80,
# 'layer4.0.conv1':0.90,
# 'layer4.0.conv2':0.95,
# 'layer4.0.shortcut.0':0.95,
# 'layer4.1.conv1':0.95,
# 'layer4.1.conv2':0.95,
# 'fc':0.80,
# }


sparsity_dict = {       #for VGG
    'backbone.conv0':0.5,
'backbone.conv1':0.6,
'backbone.conv2':0.70,
'backbone.conv3':0.7,
'backbone.conv4':0.70,
'backbone.conv5':0.80,
'backbone.conv6':0.80,
'backbone.conv7':0.80,
'fc2':0.8,
}

train_dataloader,test_dataloader=get_dataloaders(path, batch_size=64 ) # Basemodel
dense_model_accuracy=evaluate(model,test_dataloader)
print('dense_model_accuracy:',dense_model_accuracy)
pruned_model=copy.deepcopy(model)
if pruning_type=='FGP':
    isCallback=True
    pruner = FineGrainedPruner(pruned_model, sparsity_dict)
elif pruning_type=='CP':
    pruned_model = ChannelPrunner(pruned_model, sparsity_dict,select_model)
    pruner=None
    isCallback=False
else:
    print('pruning_type doesn\'t exists')
    exit

print_model(pruned_model)
print(f'The sparsity of each layer becomes')
for name, param in pruned_model.named_parameters():
    if name in sparsity_dict:
        print(f'  {name}: {get_sparsity(param):.2f}')

dense_model_size = get_model_size(model, count_nonzero_only=True)
sparse_model_size = get_model_size(pruned_model, count_nonzero_only=True)

print(f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
sparse_model_accuracy,_ = evaluate(pruned_model, test_dataloader)
print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% before fintuning")

num_finetune_epochs = 100
optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
criterion = nn.CrossEntropyLoss()

pruned_model_accuracy,best_pruned_model,accuracies,train_losses,test_losses=TrainingPrunned(pruned_model,train_dataloader,test_dataloader,criterion, optimizer, pruner,scheduler=None,num_finetune_epochs=num_finetune_epochs,isCallback=isCallback)

torch.save(best_pruned_model, f'./checkpoint/{select_model}/{pruning_type}/{select_model}_cifar_{pruning_type}_{pruned_model_accuracy}.pth')

sparse_model_accuracy,_ = evaluate(best_pruned_model, test_dataloader)

print(sparse_model_accuracy)
titel_append=f'of {pruning_type} based Pruned {select_model.title()} model'
save_path=f'./checkpoint/{select_model}/{pruning_type}/{select_model}_cifar_{pruning_type}'

plot_accuracy(accuracies,titel_append=titel_append,save_path=save_path+'_acc.png' )
plot_loss(train_losses,test_losses,titel_append=titel_append,save_path=save_path+'_loss.png')
