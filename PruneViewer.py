import pickle
import random
import torch
from DataPreprocessing import get_dataloaders
from TrainingModules import evaluate
from Utill import plot_sensitivity_scan, sensitivity_scan
from VGG import VGG
from Viewer import plot_weight_distribution  # Ensure you import your correct model architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Initialize the model
path='../dataset/cifar10'
select_model='vgg'
#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='checkpoint/vgg/vgg_cifar_93.0199966430664.pth'
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
model.to(device)

plot_weight_distribution(model)
train_dataloader,test_dataloader=get_dataloaders(path )
############# calculate sparsities (optional) #############################################
sparsities_path=f'./checkpoint/{select_model}/{select_model}_sparsities.pkl'
accuracies_path=f'./checkpoint/{select_model}/{select_model}_accuracies.pkl'

sparsities, accuracies = sensitivity_scan(
    model, test_dataloader, scan_step=0.1, scan_start=0.1, scan_end=1.0)

with open(sparsities_path, "wb") as f:
    pickle.dump(sparsities, f)

with open(accuracies_path, "wb") as f:
    pickle.dump(accuracies, f)

############################################################################################
with open(sparsities_path, "rb") as f:
    sparsities = pickle.load(f)

with open(accuracies_path, "rb") as f:
    accuracies = pickle.load(f)
dense_model_accuracy,_=evaluate(model,test_dataloader)

plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy)