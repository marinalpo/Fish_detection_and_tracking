import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
from functools import partial
assert torch.__version__.split('.')[1] == '4'
import model
"""
Usage:
srun --mem 8G --gres=gpu:1 python visualize.py --dataset csv --csv_classes /imatge/ppalau/work/Fishes/data/fishes_classes.csv --csv_val /imatge/ppalau/work/Fishes/test_image.csv --model /imatge/ppalau/work/Fishes/model_weights/0/fishes_retinanet_4.pt
"""

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file. ResNet')
    parser.add_argument('--model_retina', help='Path to retinanet model (fishes)')
    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    
    pretrained_resnet_dict = torch.load(parser.model)
    # print("Pretrained resnet")
    # print(pretrained_resnet_dict.keys())
    # print("############################################    ")
    pretrained_retinanet_model = torch.load(parser.model_retina)
    pretrained_retinanet_dict = pretrained_retinanet_model.module.state_dict()
    # print("Retinanet with resnet 50 ")
    # print(pretrained_retinanet_model.state_dict().keys())

    retinanet = model.resnet152(num_classes=2,)
    inference_retinanet_dict = retinanet.state_dict()
    
    # print("Retinanet with resnet 152 ")
    # print(retinanet.state_dict().keys())
    # print("#############################")

    print("Type of pretrained retinanet")
    print(type(pretrained_retinanet_model.module))

    print("Trying to change parameters")
    for name, param in retinanet.named_parameters():
        # print(name)
        if(param.requires_grad):
            if(("regressionModel" in name) \
                or ("classificationModel" in name) \
                or ("fpn" in name) \
                or ("layer4.2.bn3.runnig_mean" in name) \
                or ("layer4.2.bn3.running_var"in name) \
                or ("layer4.2.bn3.num_bacthes_tracked" in name)):
                
                inference_retinanet_dict[name] = pretrained_retinanet_dict[name]

            else:
                # We are in ResNet or FPN
                inference_retinanet_dict[name] = pretrained_resnet_dict[name]


    print("Parameters switched")
    retinanet.load_state_dict(inference_retinanet_dict)
	# Print model to see what resnet backbone uses
    print(retinanet)
    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()
    
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print("Scores:")
            print(scores)
            print("Classification")
            print(classification)
            print("Transformed anchors")
            print(transformed_anchors)
            print('Elapsed time: {}'.format(time.time()-st))
            # This low threshold is to check the feature proposal network (resnet)
            print("Scores")
            print(scores)
            idxs = np.where(scores>0.1)
            print("idxs")
            print(idxs)
            # TODO: figure out why plotting the raw bboxes that it gives as output does not work. If we save the image as a .npy and then visualize it is correct, but don't know why
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            #print("data")
            #print(data)
            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)
            np.save('/imatge/ppalau/work/Fishes/output_image', img)
            # x11 is not supported so we comment the imshow and save the image then plot it in local machine (cutre)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)


if __name__ == '__main__':
 main()
 print("Finished deteciton")