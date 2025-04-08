import cv2
import os
from collections import defaultdict, OrderedDict
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader,random_split,Subset,TensorDataset
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
import torchvision.models as models
import torchvision
from torchprofile import profile_macs
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
import time
from pygame import mixer
import copy
from typing import Union,List
from VGG import VGG
from TrainingModules import predict
from DataPreprocessing import test_transform
from PIL import Image


#fix the randomness
seed=0
random.seed(seed)
np.random.seed(seed)



