from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable

from models import G
from util import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='cuhk')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

netG_state_dict = torch.load(opt.model)
netG = G(opt.input_nc, opt.output_nc, opt.ngf)
netG.load_state_dict(netG_state_dict)

image_dir = "dataset/{}/test/a/".format(opt.dataset)
image_dir2 = "dataset/{}/test/a2/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
image_filenames2 = [x for x in os.listdir(image_dir2) if is_image_file(x)]

i=0
for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img2 = load_img(image_dir2 + image_filenames2[i])
    input = Variable(img).view(1, -1, 256, 256)
    input2 = Variable(img2).view(1, -1, 256, 256)

    i+=1
    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()
        input2 = input2.cuda()

    out = netG(torch.cat((input, input2),1))

    out [out > 0] = 1
    out [out <= 0] = -1

    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
