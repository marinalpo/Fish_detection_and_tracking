#import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

import numpy as np

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

#from re3_utils.tensorflow_util import tf_util
#from re3_utils.tensorflow_util.CaffeLSTMCell import CaffeLSTMCell

from constants import LSTM_SIZE
from constants import CROP_SIZE

#IMAGENET_MEAN = [123.151630838, 115.902882574, 103.062623801]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

##msra_initializer = tf.contrib.layers.variance_scaling_initializer()
##bias_initializer = tf.zeros_initializer()
##prelu_initializer = tf.constant_initializer(0.25)

class resnet_conv_layers(nn.Module):
    def __init__(self):
        super(resnet_conv_layers,self).__init__()

        # Extracting layers from the pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requiresGrad = False
        self.shallow = torch.nn.Sequential(*list(resnet.children()))[:3]
        for param in self.shallow.parameters():
            param.requiresGrad = False
        self.intermediate = torch.nn.Sequential(*list(resnet.children()))[3:6]
        for param in self.intermediate.parameters():
            param.requiresGrad = False
        self.deep = torch.nn.Sequential(*list(resnet.children()))[6:8]
        for param in self.deep.parameters():
            param.requiresGrad = False
##        self.pool = torch.nn.Sequential(*list(resnet.children()))[8]
##        for param in self.pool.parameters():
##            param.requiresGrad = False

        # Custom convolutional layers, might be replaced by pooling
        # and maybe a fully connected output layer
        #self.conv_shallow = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=11, stride=9)
        #self.conv_intermediate = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=7, stride=4)
        #self.conv_deep = torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2)
        #self.conv_final = torch.nn.Conv2d(in_channels=3*2048, out_channels=2048, kernel_size=3, stride=3)
        self.pool_shallow = torch.nn.AdaptiveAvgPool2d(output_size=(4, 8))
        self.pool_intermediate = torch.nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.pool_deep = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
    def forward(self, inputs, batch_size):

        # Normalizing the input        
        for i, m, s in zip(inputs, IMAGENET_MEAN, IMAGENET_STD):
            i.sub_(m).div_(s)
        # Shallow layers
        shallow = self.shallow(inputs) #[N, 3, K, L]->[N, 64, K/2, L/2]
        #shallow_out = self.conv_shallow(shallow) #[N,128,12,12]
        shallow_out = self.pool_shallow(shallow) #[N,64,12,24]
        #print(shallow_out.shape)

        # Intermediate layers
        intermediate = self.intermediate(shallow) #[N, 64, K/2, L/2]->[N, 512, K/(2*4), L/(2*4)]
        #intermediate_out = self.conv_intermediate(intermediate)#[N,512,6,6]
        intermediate_out = self.pool_intermediate(intermediate)#[N,512,6,6]
        #print(intermediate_out.shape)

        # Deep layers
        deep = self.deep(intermediate) #[N, 512, K/(2*4), L/(2*4)]->[N, 2048, K/(2*4*4), L/(2*4*4)]
        #deep_out = self.conv_deep(deep)#[N,2048,3,3]
        deep_out = self.pool_deep(deep)#[N,2048,3,3]
        #print(deep_out.shape)

        # One of the final layers (Not much different from the deep layers, is it needed ?)
        #pool_out = self.pool(deep_out) #[N, 2048, K/(2*4*4), L/(2*4*4)]->[N, 2048, 1, 1]
        #print(pool_out.shape)
        
##        skip_concat = torch.cat([shallow_out.reshape(392*2,1,2048),
##                                 intermediate_out.reshape(196*2,1,2048),
##                                 deep_out.reshape(49*2, 1, 2048),
##                                 pool_out.reshape(1*2,1,2048)], dim=0)

        skip_concat = torch.cat([shallow_out.reshape(batch_size*2,2048,1,1),
                                intermediate_out.reshape(batch_size*2,2048,1,1),
                                deep_out], dim=1)

        #output = self.conv_final(skip_concat)
        #reshaped = output.reshape(batch_size, 4096)
        reshaped = skip_concat.reshape(batch_size, 6*2048)
        #print(skip_concat.shape)
        #skip_concat_shape = skip_concat.get_shape().as_list()

        # Split and merge image pairs
        # (BxTx2)xHxWxC
        #pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
        #reshaped = tf_util.remove_axis(pool5_reshape, [1,3])
        return reshaped
        
class torch_net(nn.Module):
    def __init__(self):
        super(torch_net,self).__init__()
        #self.conv_layers = resnet_conv_layers()
        self.fc_input = nn.Linear(6*2*LSTM_SIZE, 2*LSTM_SIZE)
        self.lstm1 = nn.LSTM(2*LSTM_SIZE, LSTM_SIZE)
        self.lstm2 = nn.LSTM(3*LSTM_SIZE, LSTM_SIZE)
        self.fc_output = nn.Linear(LSTM_SIZE, 4)
        
    def forward(self, inputs, batch_size, prevLstmState=None, num_unrolls=1):

        # Output of the convolutional layers reshaped for the LSTMs
        #conv_out = self.conv_layers(inputs, batch_size, num_unrolls).reshape(1,batch_size,4*LSTM_SIZE)
        #conv_out = torch.rand(1, 1, 4096).type(torch.cuda.FloatTensor)
        conv_out = self.fc_input(inputs).reshape(1,batch_size,2*LSTM_SIZE)

##        if prevLstmState is not None:
        # Use the previous LSTM states given as parameters
        h01 = prevLstmState[0]
        c01 = prevLstmState[1]
        h02 = prevLstmState[2]
        c02 = prevLstmState[3]
##        else:
##            h01 = Variable(torch.rand(1, 1, 1024))
##            c01 = Variable(torch.rand(1, 1, 1024))
##            h02 = Variable(torch.rand(1, 1, 1024))
##            c02 = Variable(torch.rand(1, 1, 1024))

        #LSTM layers
        lstm1_out, state1 = self.lstm1(conv_out, (h01, c01))
        lstm2_in = torch.cat((conv_out, lstm1_out), dim=2)
        lstm2_out, state2 = self.lstm1(conv_out, (h02, c02))

        #fully connected output (1, 1, LSTM_SIZE) -> (1, 4) 
        fc_output_out = self.fc_output(lstm2_out.reshape(batch_size, LSTM_SIZE))
        
        if prevLstmState is not None:
            return fc_output_out, state1, state2
            #return fc_output_out, [0,0,0,0], [0,0,0,0]
        else:
            return fc_output_out

        #return conv_out[0][0][0:4], prevLstmState[0:2], prevLstmState[2:4]

#### Functions from the TensorFlow version, may be useful for training
##def get_var_list():
##    return tf.trainable_variables()
##
##def loss(outputs, labels):
##    with tf.variable_scope('loss'):
##        diff = tf.reduce_sum(tf.abs(outputs - labels, name='diff'), axis=1)
##        loss = tf.reduce_mean(diff, name='loss')
##
##    # L2 Loss on variables.
##    with tf.variable_scope('l2_weight_penalty'):
##        l2_weight_penalty = 0.0005 * tf.add_n([tf.nn.l2_loss(v)
##            for v in get_var_list()])
##
##    full_loss = loss + l2_weight_penalty
##
##    return full_loss, loss
##
##def training(loss, learning_rate):
##    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
##    with tf.device('/cpu:0'):
##        global_step = tf.train.create_global_step()
##    train_op = optimizer.minimize(loss, var_list=get_var_list(), global_step=global_step,
##        colocate_gradients_with_ops=True)
##    return train_op

####For testing the network directly
##testResnet = resnet_conv_layers()
##input = torch.autograd.Variable(torch.rand(2, 3, 224, 224).type(torch.cuda.FloatTensor))
##lstmState = [torch.autograd.Variable(torch.rand(1, 2, LSTM_SIZE).type(torch.cuda.FloatTensor)) for _ in range(4)]
##testResnet.forward(input, 1, 1)
##torchnet = torch_net().cuda()
###input = Variable(torch.rand(196, 1, 2048))
##print(torchnet(input, 1, lstmState))
