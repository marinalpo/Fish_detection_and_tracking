from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import G, D, weights_init
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy import ndimage  #chafik
from scipy.misc import imread, imresize, imsave #chafik
from util import is_image_file, load_img, save_img, convert_img, unconvert_img #chafik
from torchviz import make_dot

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=float, default=100, help='weight on L1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset)
test_set = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
netG = G(opt.input_nc, opt.output_nc, opt.ngf)
netG.apply(weights_init)
netD = D(opt.input_nc, opt.output_nc, opt.ndf)
netD.apply(weights_init)
netDL = D(opt.input_nc, opt.output_nc, opt.ndf)
netDL.apply(weights_init)

criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

real_A = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_A2 = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_B = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
label = torch.FloatTensor(opt.batchSize)
label_DL = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD = netD.cuda()
    netDL = netDL.cuda()
    netG = netG.cuda()
    criterion = criterion.cuda()
    criterion_l1 = criterion_l1.cuda()
    criterion_mse = criterion_mse.cuda()
    real_A = real_A.cuda()
    real_A2 = real_A2.cuda()
    real_B = real_B.cuda()
    label = label.cuda()
    label_DL = label_DL.cuda()


real_A = Variable(real_A)
real_A2 = Variable(real_A2)
real_B = Variable(real_B)
label = Variable(label)
label_DL = Variable(label_DL)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerDL = optim.Adam(netDL.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


array_err_g = np.zeros(100)
array_psnr = np.zeros(100)

def dice_loss(input,target): 
    num=input*target 
    num=torch.sum(num,dim=2) 
    num=torch.sum(num,dim=2) 
    den1=input*input 
    den1=torch.sum(den1,dim=2) 
    den1=torch.sum(den1,dim=2) 
    den2=target*target
    den2=torch.sum(den2,dim=2) 
    den2=torch.sum(den2,dim=2) 
    dice=2*(num/(den1+den2)) 
    dice_total=1-1*torch.sum(dice)/3#dice.size(0) #divide by batchsize  
    return dice_total

def train(epoch):
    avg_err_g = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train with real
        netD.volatile = False
        netD.zero_grad()
        real_a_cpu, real_a2_cpu, real_b_cpu = batch[0], batch[1] , batch[2]
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_A2.data.resize_(real_a2_cpu.size()).copy_(real_a2_cpu)
        real_B.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

        output = netD(torch.cat((real_A, real_A2, real_B), 1))
        label.data.resize_(output.size()).fill_(real_label)
        err_d_real = criterion(output, label)
        err_d_real.backward()
        d_x_y = output.data.mean()
      
        # train with fake
        fake_b = netG(torch.cat((real_A, real_A2),1))
        output = netD(torch.cat((real_A, real_A2, fake_b.detach()), 1))
        label.data.resize_(output.size()).fill_(fake_label)
        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        d_x_gx = output.data.mean()

        err_d = (err_d_real + err_d_fake) / 2.0
        optimizerD.step()

        ############################
        #chafik
        ############################
        
        filename=batch[3]   
          
        # load training image  (version1)
        #real_B_numpy = imread(train_set.b_path+'/'+filename[0])
        #real_B_numpy = imresize(real_B_numpy, (256, 256))
        #real_A_numpy = imread(train_set.a_path+'/'+filename[0])
        #real_A_numpy = imresize(real_A_numpy, (256, 256))

        # load training image  (version2)
        real_B_numpy = unconvert_img(real_B.cpu().data[0])
        real_A_numpy = unconvert_img(real_A.cpu().data[0])
        real_A2_numpy = unconvert_img(real_A2.cpu().data[0])
        fake_b_numpy = unconvert_img(fake_b.detach().cpu().data[0])
        
	# find connected components
        labeled, nr_objects = ndimage.label(real_B_numpy>0)
        
        # train local discriminator DL with each object      
        for l in range(1, nr_objects+1):
      
            loc = ndimage.find_objects(labeled==l)[0]
          
            object_fake_b_numpy = fake_b_numpy[loc]
            object_real_B_numpy = real_B_numpy[loc]
            object_real_A_numpy = real_A_numpy[loc]
            object_real_A2_numpy = real_A2_numpy[loc]

            #imsave('./tmp/'+filename[0]+'_'+str(l)+'_real_B.png', object_real_B_numpy)
            #imsave('./tmp/'+filename[0]+'_'+str(l)+'_fake_b.png', object_fake_b_numpy)
            #imsave('./tmp/'+filename[0]+'_'+str(l)+'_real_A.png', object_real_A_numpy)
            #imsave('./tmp/'+filename[0]+'_'+str(l)+'_real_A2.png', object_real_A2_numpy)
               
            object_real_B = convert_img(object_real_B_numpy)
            object_real_B = Variable(object_real_B).view(1, -1, 256, 256)

            object_fake_b = convert_img(object_fake_b_numpy)
            object_fake_b = Variable(object_fake_b).view(1, -1, 256, 256)

            object_real_A = convert_img(object_real_A_numpy)
            object_real_A = Variable(object_real_A).view(1, -1, 256, 256)


            object_real_A2 = convert_img(object_real_A2_numpy)
            object_real_A2 = Variable(object_real_A2).view(1, -1, 256, 256)

            if opt.cuda:
                object_fake_b = object_fake_b.cuda()
                object_real_B = object_real_B.cuda()
                object_real_A = object_real_A.cuda()
                object_real_A2 = object_real_A2.cuda()

            # train local D
            netDL.volatile = False
            netDL.zero_grad()

            # train with real       
            output_DL = netDL(torch.cat((object_real_A, object_real_A2 , object_real_B), 1))
            label_DL.data.resize_(output_DL.size()).fill_(real_label)
            err_d_real_DL = criterion(output_DL, label_DL)
            err_d_real_DL.backward()

            # train with fake
            output_DL = netDL(torch.cat((object_real_A, object_real_A2 ,object_fake_b), 1))
            label_DL.data.resize_(output_DL.size()).fill_(fake_label)
            err_d_fake_DL = criterion(output_DL, label_DL)
            err_d_fake_DL.backward()

            optimizerDL.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ###########################
        netG.zero_grad()
        netD.volatile = True

        #loss of global D
        output = netD(torch.cat((real_A, real_A2, fake_b), 1))        
        label.data.resize_(output.size()).fill_(real_label)


        #loss of discriminator DL
        #here we need to crop tensor
        err_d_DL = 0
        list_output_DL = []
        list_label_DL = []

        i = 0
        for l in range(1, nr_objects+1):

            loc = ndimage.find_objects(labeled==l)[0]

            object_fake_b_new = fake_b[:, :, loc[0].start:loc[0].stop, loc[1].start:loc[1].stop]

            object_real_B_new = real_B[:, :, loc[0].start:loc[0].stop, loc[1].start:loc[1].stop]
            object_real_A_new = real_A[:, :, loc[0].start:loc[0].stop, loc[1].start:loc[1].stop]
            object_real_A2_new = real_A2[:, :, loc[0].start:loc[0].stop, loc[1].start:loc[1].stop]

            #upsampling
            object_fake_b_new=F.upsample(object_fake_b_new, size=(256,256), mode='bilinear')

            object_real_B_new=F.upsample(object_real_B_new, size=(256,256), mode='bilinear')
            object_real_A_new=F.upsample(object_real_A_new, size=(256,256), mode='bilinear')
            object_real_A2_new=F.upsample(object_real_A2_new, size=(256,256), mode='bilinear')

            #imsave('./tmp2/'+filename[0]+'_'+str(l)+'_fake_b.png', unconvert_img(object_fake_b_new.cpu().data[0]))
            #imsave('./tmp2/'+filename[0]+'_'+str(l)+'_real_B.png', unconvert_img(object_real_B_new.cpu().data[0]))
            #imsave('./tmp2/'+filename[0]+'_'+str(l)+'_real_A2.png', unconvert_img(object_real_A2_new.cpu().data[0]))
                             
            # train with fake
            netDL.volatile = True
            output_DL = netDL(torch.cat((object_real_A_new, object_real_A2_new, object_fake_b_new), 1))

            list_output_DL.append(output_DL)
          
            label_DL.data.resize_(output_DL.size()).fill_(real_label)

            list_label_DL.append(label_DL)
                    
            err_d_DL = err_d_DL + criterion(list_output_DL[i], list_label_DL[i]) 

            i = i + 1

        # Here we can choose how to compute the total loss

        ################
        err_g = err_d_DL
        ################
        #err_g = criterion(output, label)
        ################
        #err_g = criterion_l1(fake_b, real_B)
        ################
        #err_g = criterion(output, label) + 50 * err_d_DL + 100 * dice_loss(fake_b, real_B)
      
        #if (nr_objects > 0):
        err_g.backward()
        d_x_gx_2 = output.data.mean()
        optimizerG.step()
        avg_err_g = avg_err_g + err_g.data[0]
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}".format(epoch, iteration, len(training_data_loader), err_d.data[0], err_g.data[0], d_x_y, d_x_gx, d_x_gx_2))

    avg_err_g =  avg_err_g / iteration
    array_err_g[epoch-1] = avg_err_g
    np.savetxt('curves/array_err_g.out', array_err_g)

def test(epoch):
    avg_psnr = 0
    for batch in testing_data_loader:
        input, input2, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if opt.cuda:
            input = input.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        prediction = netG(torch.cat((input, input2),1))
        mse = criterion_mse(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    avg_psnr = avg_psnr/ len(testing_data_loader)
    array_psnr[epoch-1] = avg_psnr
    np.savetxt('curves/array_psnr.out', array_psnr)


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(netG.state_dict(), net_g_model_out_path)
    torch.save(netD.state_dict(), net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 50 ==0:
        checkpoint(epoch)
