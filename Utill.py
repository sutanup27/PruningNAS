import copy
import math
from torch import nn
import time
from typing import List, Union
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from TrainingModules import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_url(url, model_dir='.', overwrite=False):
    import os, sys, ssl
    from urllib.request import urlretrieve
    ssl._create_default_https_context = ssl._create_unverified_context
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, 'download.lock'))
        sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
        return None


def magnitude_based_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(num_elements*sparsity)
    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)
    # Step 3: calculate the pruning threshold
    threshold,ind=  torch.kthvalue(torch.abs(tensor.view(-1)), num_zeros)
    # Step 4: calculate the pruning mask
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance.gt(threshold)
    ##################### YOUR CODE ENDS HERE #######################

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


def fine_grained_prune(tensor: torch.Tensor, sparsity : float, prune_type="magnitude_based"):
    if prune_type=="magnitude_based" :
        return magnitude_based_prune(tensor,sparsity)
    else:
        print("Wrong prune type is passed")
        return tensor
        

    
def get_labels_preds(model, dataloader,criterion):
  all_labels = []
  all_preds = []
  all_outputs=[]
  for inputs, targets in dataloader:
    preds1=model(inputs.to(device))
    preds=preds1.argmax(dim=1)
    preds = preds.cpu().numpy()  # Convert to numpy array for sklearn
    all_outputs.append(preds1.cpu().detach().numpy())
    all_preds.append(preds)
    all_labels.append(targets.numpy())  # Convert to numpy array for sklearn
    loss = criterion(preds1, targets.cuda())

  all_preds=[item for sublist in all_preds for item in sublist]
  all_labels = [item for sublist in all_labels for item in sublist]
  all_outputs = [item for sublist in all_outputs for item in sublist]

  return all_labels, all_preds, all_outputs, loss

@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                          in model.named_modules() if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear)]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.weight.detach().clone()
        accuracy = []
        for sparsity in sparsities:
            fine_grained_prune(param.weight.detach(), sparsity=sparsity)
            acc,_ = evaluate(model, dataloader, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.weight.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
    return sparsities, accuracies



def plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy,save_path=None):
    layer_count=0
    for name, param in model.named_modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            layer_count=   layer_count+1
    col= round(3*math.sqrt(layer_count/12.0))
    row= round(layer_count/col)
    if col*row<layer_count:
        col=col+1

    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(col, row, figsize=(50,50),constrained_layout=True)
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.1, stop=1.0, step=0.1))
            ax.set_ylim(80, 100)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend([
                'accuracy after pruning',
                f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
            ])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    #fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if save_path is None:
        plt.show()
    else:
        plt.savefig("sensitivity_scan.png") 
    plt.close()

def recover_model(PATH,model):
        cp = torch.load(download_url(PATH), map_location="cpu")
        return model.load_state_dict(cp)


def print_model(model):
    for name, param in model.named_modules():
        if isinstance(param,nn.Conv2d) or isinstance(param,nn.Linear):
            print(name, param.weight.shape)


#   can directly leads to model size reduction and speed up.
@torch.no_grad()
def measure_latency(main_model,dummy_input, n_warmup=20, n_test=100, d='cpu'):
    model=copy.deepcopy(main_model)
    model.to(d)
    input=copy.deepcopy(dummy_input)
    input=input.to(d)
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    ##################### YOUR CODE STARTS HERE #####################
    return round(channels*(1-prune_ratio))
    ##################### YOUR CODE ENDS HERE #####################
