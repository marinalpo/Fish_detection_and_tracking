# pix2pix-pytorch

PyTorch implementation of [BScGAN](https://ieeexplore.ieee.org/document/8451603).

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

The examples from the paper: 

the input img size is width = "512" height = "512" 

## Prerequisites

+ Linux
+ Python with numpy
+ NVIDIA GPU + > CUDA 8.0 + > CuDNNv5.1
+ pytorch
+ torchvision

## dataset
+ Train folder contains 
	++ a folder for input images
	++ a2 folder for input references (background)
	++ b folder for targert (forgtound ground-truth)
+ Test folder contains 
	++ a folder for input images
	++ a2 folder for input references (background)
	++ b folder for targert (forgtound ground-truth for evaluting the model)

## Getting Started

+ Train the model:

    python train.py --dataset $dataset --nEpochs 200 --cuda

+ Test the model:

    python test.py  --dataset $dataset --model checkpoint/$dataset/netG_model_epoch_200.pth --cuda



