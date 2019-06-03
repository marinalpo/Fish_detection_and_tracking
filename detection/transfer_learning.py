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

"""srun --mem 8G --gres=gpu:1,gmem:10G python transfer_learning.py  --learning_mode 1 --model /imatge/ppalau/sespeixos/coco_resnet_50_map_0_335_state_dict.pt --csv_train /imatge/ppalau/work/Fishes/data/train.csv  --classes /imatge/ppalau/work/Fishes/data/fishes_classes.csv --val /imatge/ppalau/work/Fishes/data/val.csv"""
"""
I tried to train the network importing the whole model but it did not work. Now in this script I will try to load the state_dict weights and retrain them.
""" 

# To try more than two
class LearningMode(IntEnum):
    FINETUNING = 1
    FEATURE_EXTRACTOR = 2

def train_model(retinanet, dataset_train, dataset_val, dataloader_train, dataloader_val):
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.module.freeze_bn()
    epochs = 500
    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.training = True
        retinanet.module.freeze_bn()
        mAPs = {}
        train_summary = []
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
                # print("weights at output layer = ")
                # print(retinanet.module.classificationModel.output.bias)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        print('Evaluating dataset')
        mAP,p_r = csv_eval.evaluate(dataset_val, retinanet)
        train_summary.append((mAP,p_r))
        print(train_summary)
        print(mAP)
        scheduler.step(np.mean(epoch_loss))    
        f_score = 2 * (p_r[0]*p_r[1])/(p_r[0] + p_r[1])
        torch.save(retinanet, '/imatge/ppalau/work/Fishes/model_weights/4/fishes_retinanet_epoch_{}_F_{}.pt'.format(epoch_num, f_score))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt'.format(epoch_num))



def main(args=None):
    # ARGUMENT PARSE
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--learning_mode', help = "0: FINETUNING, 1: FEATURE_EXTRACTOR")
    parser.add_argument('--model', help = 'Path to file containing pretrained weights (state_dict)')
    parser.add_argument('--csv_train', help ='Path to file containing training annotations (output from xml_to_csv.py)')
    parser.add_argument('--classes', help ='Path to file containing class list')
    parser.add_argument('--val', help ='Path to file containing validation annotations (output from xml_to_csv.py)')
    args = parser.parse_args()
    
    learning_mode = int(args.learning_mode)
    print("We are going to start the training...")
    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the datasets
    dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CSVDataset(train_file=args.val, class_list=args.classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    
    # Define dataloaders for each dataset
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=5, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=5, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    
    # Load pretrained model
    print("Number of classes = " + str(dataset_train.num_classes()))
    retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
    
    pretrained_dict = torch.load(args.model)
    model_dict = retinanet.state_dict()

    print("################# Pretrained ######################")
    #print(pretrained_dict['classificationModel.output.weight'])
    #print(pretrained_dict['classificationModel.output.bias'])
    print("######## Your retinanet")
    #print(model_dict['classificationModel.output.weight'])
    #print(model_dict['classificationModel.output.bias'])
    
    # Our model differs just in the last layer of the classification model,so we initialize them with the default values
    pretrained_dict['classificationModel.output.weight'] = model_dict['classificationModel.output.weight'] 
    pretrained_dict['classificationModel.output.bias'] = model_dict['classificationModel.output.bias'] 
    
    retinanet.load_state_dict(pretrained_dict)
    print("Retinanet model before dataparallel:")
    print(type(retinanet))
    retinanet.dump_patches = True
    # When we work with a DataParallel object we must access the model using model.module
    retinanet = torch.nn.DataParallel(retinanet)
    retinanet.cuda()

    print("Retinanet model after dataparallel:")
    print(type(retinanet.module))
    print("Retinanet's classification model state_dict:")
    print(retinanet.module.classificationModel.state_dict().keys())


    print("################################################3")
   
    

    if(learning_mode == LearningMode.FINETUNING):
        # Begin with pretrained model and train it all
        print("Model load successfully, finetuning selcted")
        # Train the whole model
        train_model(retinanet, dataset_train, dataset_val, dataloader_train,dataloader_val)
    elif(learning_mode == LearningMode.FEATURE_EXTRACTOR):
        print("Model load successfully, feature extraction selected")
        # Freeze ResNet and FPN layers and train the rest 
        for name, param in retinanet.module.named_parameters():
            if(param.requires_grad):
                print(name)
                if(not (("regressionModel.output" in name) or ("regressionModel.conv4" in name) or ("classificationModel.output" in name) or ("classificationModel.conv3" in name) or ("classificationModel.conv4" in name))):
                    param.requires_grad = False
                else:
                    print("This layer will be retrained: " + str(name))
        train_model(retinanet, dataset_train, dataset_val, dataloader_train,dataloader_val)

if __name__ == '__main__':
    main()