import numpy as np
import random
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slows training but ensures reproducibility
mean =[-0.3308, -0.3385, -0.3125]   #[0.4914, 0.4822, 0.4465]
std = [1.2540, 1.2479, 1.2031]      #[0.2023, 0.1994, 0.2010]

image_size = 32
train_transform=Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std),  # ImageNet normalization
    ])
test_transform= Compose([
        ToTensor(),
        Normalize(mean, std),  # ImageNet normalization
    ])
 
def get_datasets(path,train_transform=train_transform,test_transform=test_transform,train_test_val_pecentage=[0.80, 0.20]):
    transform={}
    transform["train"], transform["test"]=train_transform,test_transform
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root=path,
            train=(split == "train"),
            download=True,
            transform=transform[split],)
    return dataset["train"],dataset["test"]



def get_dataloaders(path,train_transform=train_transform,test_transform=test_transform, train_test_val_pecentage=[0.80, 0.20], batch_size=512):
    set_seed(0)
    dataset={}
    dataset["train"], dataset["test"]=get_datasets(path, train_transform, test_transform, train_test_val_pecentage)
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )
    return dataloader['train'],dataloader['test']


def calculate_mean_std(loader):
    # Load dataset without normalization
    transform = transforms.Compose([
        transforms.ToTensor()  # Important: no normalization here
    ])

    # CIFAR-10 training set
    mean = 0.0
    std = 0.0
    nb_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # get the batch size
        images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
        mean += images.mean(2).sum(0)  # mean over pixels, sum over batch
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

# path='../dataset/cifar10'

# train_dataloader,test_dataloader=get_dataloaders(path)

# mean,std=calculate_mean_std(train_dataloader)
# print("Calculated Mean:", mean)
# print("Calculated Std:", std)
