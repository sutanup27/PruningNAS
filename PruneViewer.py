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
select_model='Vgg-16'
prune_type='CP'
#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='checkpoint/Vgg-16/Vgg-16_cifar_93.0199966430664.pth'
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
model.to(device)

train_dataloader,test_dataloader=get_dataloaders(path )
############# calculate sparsities (optional) #############################################
sparsities_path=f'./checkpoint/{select_model}/{prune_type}/{select_model}_sparsities.pkl'
accuracies_path=f'./checkpoint/{select_model}/{prune_type}/{select_model}_accuracies.pkl'

sparsities, accuracies,names = sensitivity_scan(
    model, test_dataloader, scan_step=0.1, scan_start=0.1, scan_end=1.0,prune_type=prune_type,select_model=select_model)

with open(sparsities_path, "wb") as f:
    pickle.dump(sparsities, f)

with open(accuracies_path, "wb") as f:
    pickle.dump((accuracies,names), f)

############################################################################################
with open(sparsities_path, "rb") as f:
    sparsities = pickle.load(f)

with open(accuracies_path, "rb") as f:
    accuracies,names = pickle.load(f)
dense_model_accuracy,_=evaluate(model,test_dataloader)

save_image_path1=f'./checkpoint/{select_model}/{prune_type}/param_plot/{select_model}_paramplot_{prune_type}'
save_image_path2=f'./checkpoint/{select_model}/{prune_type}/sensitivity_curves/{select_model}_sensitivity_{prune_type}'
plot_weight_distribution(model,names,save_path=save_image_path1)
plot_sensitivity_scan( names, sparsities, accuracies, dense_model_accuracy,save_image_path2)