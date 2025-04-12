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
model = VGG()  # Replace with your actual model class
path='../dataset/cifar10'
#model_path='./checkpoint/vgg_mrl_99.51375579833984.pth'
model_path='vgg_mrl_99.09.pth'
# Load the saved state_dict correctly
state_dict = torch.load(model_path, map_location=torch.device(device),weights_only=True)  # Use 'cpu' if necessary
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
model.to(device)

plot_weight_distribution(model)
train_dataloader,test_dataloader=get_dataloaders(path,batch_size=64 )
############# calculate sparsities (optional) #############################################

# sparsities, accuracies = sensitivity_scan(
#     model, test_dataloader, scan_step=0.1, scan_start=0.1, scan_end=1.0)

# with open("sparsities.pkl", "wb") as f:
#     pickle.dump(sparsities, f)

# with open("accuracies.pkl", "wb") as f:
#     pickle.dump(accuracies, f)

############################################################################################
with open("sparsities.pkl", "rb") as f:
    sparsities = pickle.load(f)

with open("accuracies.pkl", "rb") as f:
    accuracies = pickle.load(f)
dense_model_accuracy,_=evaluate(model,test_dataloader)

plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy)