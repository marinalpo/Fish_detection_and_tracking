import time
import os
import copy
import argparse
import pdb
import collections
import sys
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'
import pickle
from functools import partial


# To try more than two
class LearningMode(Enum):
    FINETUNING = 0
    FEATURE_EXTRACTOR = 1

def train_model():
    print("")
def main(args=None):
    # ARGUMENT PARSE
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--learning_mode', help = "0: FINETUNING, 1: FEATURE_EXTRACTOR")
    parser.add_argument('--model', help = 'Path to file containing pretrained weights')
    parser.add_argument('--csv_train', help ='Path to file containing training annotations (output from xml_to_csv.py)')
    parser.add_argument('--classes', help ='Path to file containing class list')
    parser.add_argument('--val', help ='Path to file containing validation annotations (output from xml_to_csv.py)')
    args = parser.parse_args()
    print(args)
    learning_mode = args.learning_mode

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the datasets
    dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CSVDataset(train_file=args.val, class_list=args.classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    
    # Define dataloaders for each dataset
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    retinanet = torch.load(args.model, pickle_module=pickle)

    if(learning_mode == LearningMode.FINETUNING):
        # Begin with pretrained model and train it all
        print("Model load correctly, finetuning selcted")
    elif(learning_mode == LearningMode.FEATURE_EXTRACTOR):
        # Freeze some layers (and networks, for example ResNet and FPN) and train the rest 
        print("Model load correctly, feature extraction selected")
        
if __name__ == '__main__':
    main()