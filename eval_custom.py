from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np

from custom_dataloader import FaceDataLoader
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere

data_root = "./data/code (2)/deep_learning_data_x"
data_loader = FaceDataLoader(batch_size=32, num_workers=4, path=data_root, txt_path="./")
train_loader, test_loader = data_loader.run()

def fine_tune(trian_loader, model):
    pass

def save_gallery(test_loader, model):
    pass