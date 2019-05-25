import time
import os
import copy
import argparse
import pdb
import collections
import sys
from enum import IntEnum
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

"""srun --mem 8G --gres=gpu:1,gmem:10G python transfer_learning.py  --learning_mode 0 --model /imatge/ppalau/work/Fishes/coco_resnet_50_map_0_335.pt --csv_train /imatge/ppalau/work/Fishes/test_image.csv  --classes /imatge/ppalau/work/Fishes/classes_mappings.csv --val /imatge/ppalau/work/Fishes/csv_val.csv"""
"""
/work/bdorra/annotations/ if you need data. The corresponding images are in /work/morros/fish/video_segments/
""" 

# TODO: try loading a model with resnet 152 

# To try more than two
class LearningMode(IntEnum):
    FINETUNING = 1
    FEATURE_EXTRACTOR = 2

def train_model(retinanet, dataset_train, dataset_val, dataloader_train, dataloader_val):
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.module.freeze_bn()
    epochs = 120
    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()
        
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss
                
                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
            print('Evaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        
        scheduler.step(np.mean(epoch_loss))    

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt'.format(epoch_num))



def main(args=None):
    # ARGUMENT PARSE
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--learning_mode', help = "0: FINETUNING, 1: FEATURE_EXTRACTOR")
    parser.add_argument('--model', help = 'Path to file containing pretrained weights')
    parser.add_argument('--csv_train', help ='Path to file containing training annotations (output from xml_to_csv.py)')
    parser.add_argument('--classes', help ='Path to file containing class list')
    parser.add_argument('--val', help ='Path to file containing validation annotations (output from xml_to_csv.py)')
    args = parser.parse_args()
    
    learning_mode = int(args.learning_mode)
    
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
    # Load pretrained model
    retinanet = torch.load(args.model, pickle_module=pickle)
    print("Retinanet model:")

    print(retinanet.layer1)
    
    for param in retinanet.layer1.parameters():
        print(param.requires_grad)
    retinanet.train(True)


    if(learning_mode == LearningMode.FINETUNING):
        # Begin with pretrained model and train it all
        print("Model load successfully, finetuning selcted")
        # Train the whole model
        train_model(retinanet, dataset_train, dataset_val, dataloader_train,dataloader_val)
    elif(learning_mode == LearningMode.FEATURE_EXTRACTOR):
        print("Model load successfully, feature extraction selected")
        # Freeze ResNet and FPN layers and train the rest 
        
        train_model(retinanet, dataset_train, dataset_val, dataloader_train,dataloader_val)

if __name__ == '__main__':
    main()