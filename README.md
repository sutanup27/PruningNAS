
# PruningNAS

PruningNAS is a compact research codebase for training and pruning CNNs on CIFAR-10, with a focus on structured (channel) pruning and fine-grained pruning. It includes ResNet (basic and bottleneck) and VGG models, sensitivity scans, pruning utilities, and visualization helpers for pruning analysis.

## Highlights

- CIFAR-10 training with ResNet-18/34/50/101/152 and VGG-16
- Channel pruning (CP) and fine-grained pruning (FGP)
- Sensitivity scanning with accuracy curves and parameter distributions
- Checkpoint management and plotting utilities

## Repository Layout

- [PruningNAS/Train.py](Train.py) - train a dense model
- [PruningNAS/PruneTrain.py](PruneTrain.py) - prune and finetune a model
- [PruningNAS/PruneEvaluator.py](PruneEvaluator.py) - sensitivity scan and plots
- [PruningNAS/DoThings.py](DoThings.py) - small experiments and quick checks
- [PruningNAS/Models](Models) - model definitions (ResNet, VGG)
- [PruningNAS/Utills](Utills) - pruning, evaluation, training, plotting
- [PruningNAS/DataProcess](DataProcess) - CIFAR-10 dataloading
- [PruningNAS/checkpoint](checkpoint) - saved weights and plots
- [PruningNAS/Information.txt](Information.txt) - example configs and sparsity tables

## Setup

Create and activate a virtual environment, then install dependencies from [PruningNAS/requirement.txt](requirement.txt):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r PruningNAS\requirement.txt
```

## Dataset

The code expects CIFAR-10 under:

```
dataset/cifar10/cifar-10-batches-py/
```

If you already have CIFAR-10 downloaded, place it there. Otherwise, update the dataset path in [PruningNAS/Train.py](Train.py), [PruningNAS/PruneTrain.py](PruneTrain.py), and [PruningNAS/PruneEvaluator.py](PruneEvaluator.py).

## Quickstart

### 1) Train a dense model

Edit `select_model` and training hyperparameters in [PruningNAS/Train.py](Train.py), then run:

```powershell
python -m PruningNAS.Train
```

Checkpoints are saved under:

```
PruningNAS/checkpoint/<ModelName>/
```

### 2) Sensitivity scan (CP or FGP)

Update `select_model`, `prune_type`, and `model_path` in [PruningNAS/PruneEvaluator.py](PruneEvaluator.py), then run:

```powershell
python -m PruningNAS.PruneEvaluator
```

This writes sensitivity curves and parameter plots to:

```
PruningNAS/checkpoint/<ModelName>/<PruneType>/
```

### 3) Prune and finetune

Edit `select_model`, `pruning_type`, and `sparsity_dict` in [PruningNAS/PruneTrain.py](PruneTrain.py), then run:

```powershell
python -m PruningNAS.PruneTrain
```

The pruned checkpoint is saved under:

```
PruningNAS/checkpoint/<ModelName>/<PruneType>/
```

## Model and Pruning Notes

- ResNet basic blocks are in [PruningNAS/Models/ResNetBasic.py](Models/ResNetBasic.py)
- ResNet bottleneck blocks are in [PruningNAS/Models/ResNetBottleNeck.py](Models/ResNetBottleNeck.py)
- Pruning helpers live in [PruningNAS/Utills/PrunUtillCP.py](Utills/PrunUtillCP.py) and [PruningNAS/Utills/PrunUtillFGP.py](Utills/PrunUtillFGP.py)
- Example sparsity configurations for VGG-16 and ResNet are listed in [PruningNAS/Information.txt](Information.txt)

## Repro Tips

- Set `seed` in the training/pruning scripts for repeatable runs.
- Keep model and pruning settings aligned with your checkpoint names to avoid confusion.

