import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from sub_modules.component_block import *
from sub_modules.ConvLSTM_pytorch.convlstm import *

class Guide_pixel_offset(nn.Module):

    def __init__(self, opt):
        super().__init__()

        channel = opt.channel
        self.k_size = 3
        self.per_pix_k_size = 3
        self.recurrent = opt.Recurrent_times
        self.ConvLSTM = nn.ModuleDict()
        self.ConvLSTM_per_pix_w = nn.ModuleDict()
        self.leaky = nn.LeakyReLU(0.2)

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Mapping Network
        self.Mapping_Network = nn.Sequential(
                            nn.Conv2d(channel*3, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            Res_Block('relu', channel),
                            Res_Block('relu', channel),
                            Res_Block('relu', channel),
                            )

        self.Mapping_Network_motion_prior = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            #Base_Res_Block(opt, channel),
                            Base_Res_Block(opt, channel),
                            nn.Conv2d(channel, 3 * self.per_pix_k_size ** 2 , kernel_size = 3 , stride = 1, padding = 1),
                            Base_Res_Block(opt, 3 * self.per_pix_k_size ** 2 ),
                            )    

        for i in range(self.recurrent - 1):
            level = f'level{i}'
            self.ConvLSTM[level] = ConvLSTM(input_dim = 3 * self.per_pix_k_size ** 2, hidden_dim=3 * self.per_pix_k_size ** 2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

        

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)

        images_lv1 = x

        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = self.Patch_lv3_FE(images_lv3_1.contiguous())
        feature_lv3_2 = self.Patch_lv3_FE(images_lv3_2.contiguous())
        feature_lv3_3 = self.Patch_lv3_FE(images_lv3_3.contiguous())
        feature_lv3_4 = self.Patch_lv3_FE(images_lv3_4.contiguous())
    
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)

        
        feature_lv2_1 = self.Patch_lv2_FE(images_lv2_1.contiguous())
        feature_lv2_2 = self.Patch_lv2_FE(images_lv2_2.contiguous())

        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)

        blur_feature = self.Mapping_Network(torch.cat((self.Patch_lv1_FE(x), feature_lv2, feature_lv3), 1))

        # get motion prior
        motion_prior = self.Mapping_Network_motion_prior(blur_feature)
        self.motion_prior_list = [motion_prior]
        
        hidden_state = None
        for i in range(self.recurrent - 1):
            level = f'level{i}'
            layer_outputs, hidden_state = self.ConvLSTM[level](torch.unsqueeze(self.motion_prior_list[i], 1), hidden_state)
            self.motion_prior_list.append(torch.squeeze(layer_outputs[0], 1))

        self.feature_list = self.motion_prior_list
        

        #self.feature_list = [motion_prior]

        # motion_prior size : [batch, 3 * self.per_pix_k_size **2, H, W]
        return {'motion_prior': self.motion_prior_list}



class Patch_Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        channel = opt.channel
        self.k_size = 3
        self.per_pix_k_size = 3
        self.recurrent = opt.Recurrent_times
        self.ConvLSTM = nn.ModuleDict()
        self.ConvLSTM_per_pix_w = nn.ModuleDict()
        self.leaky = nn.LeakyReLU(0.2)

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Mapping Network
        self.Mapping_Network = nn.Sequential(
                            nn.Conv2d(channel*3, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            Res_Block('relu', channel),
                            Res_Block('relu', channel),
                            Res_Block('relu', channel),
                            Res_Block('relu', channel),
                            Res_Block('relu', channel),
                            )

        self.Mapping_Network_motion_prior = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            #get_norm_layer('relu', channel),
                            Res_Block('relu', channel),
                            #Res_Block('relu', channel),
                            #Base_Res_Block(opt, channel),
                            #Base_Res_Block(opt, channel),
                            nn.Conv2d(channel, 3 * self.per_pix_k_size ** 2 , kernel_size = 3 , stride = 1, padding = 1),
                            #get_norm_layer('relu', 3 * self.per_pix_k_size ** 2),
                            Res_Block('relu', 3 * self.per_pix_k_size ** 2),
                            #Res_Block('relu', 3 * self.per_pix_k_size ** 2),
                            #Base_Res_Block(opt, 3 * self.per_pix_k_size ** 2),
                            #Base_Res_Block(opt, 3 * self.per_pix_k_size ** 2),
                            )    

        for i in range(self.recurrent - 1):
            level = f'level{i}'
            self.ConvLSTM[level] = ConvLSTM(input_dim = 3 * self.per_pix_k_size ** 2, hidden_dim=3 * self.per_pix_k_size ** 2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)


        self.Mapping_Network_DCN = nn.Sequential(
                            #DCN(channel, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            Base_Res_Block(opt, channel),
                            #Base_Res_Block(opt, channel),
                            nn.Conv2d(channel, channel*2, kernel_size=(3, 3), stride = 2, padding = 1),
                            Base_Res_Block(opt, channel*2),
                            #Base_Res_Block(opt, channel*2),
                            nn.Conv2d(channel*2, channel*4, kernel_size=(3, 3), stride = 2, padding = 1),
                            #Base_Res_Block(opt, channel*4),
                            Base_Res_Block(opt, channel*4),)
        

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)

        images_lv1 = x

        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = self.Patch_lv3_FE(images_lv3_1.contiguous())
        feature_lv3_2 = self.Patch_lv3_FE(images_lv3_2.contiguous())
        feature_lv3_3 = self.Patch_lv3_FE(images_lv3_3.contiguous())
        feature_lv3_4 = self.Patch_lv3_FE(images_lv3_4.contiguous())
    
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)

        
        feature_lv2_1 = self.Patch_lv2_FE(images_lv2_1.contiguous())
        feature_lv2_2 = self.Patch_lv2_FE(images_lv2_2.contiguous())

        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)

        blur_feature = self.Mapping_Network(torch.cat((self.Patch_lv1_FE(x), feature_lv2, feature_lv3), 1))

        # get motion prior
        motion_prior = self.Mapping_Network_motion_prior(blur_feature)
        self.motion_prior_list = [motion_prior]
        
        hidden_state = None
        for i in range(self.recurrent - 1):
            level = f'level{i}'
            layer_outputs, hidden_state = self.ConvLSTM[level](torch.unsqueeze(self.motion_prior_list[i], 1), hidden_state)
            self.motion_prior_list.append(torch.squeeze(layer_outputs[0], 1))

        self.feature_list = self.motion_prior_list

        content_feature = self.Mapping_Network_DCN(blur_feature)
        

        #self.feature_list = [motion_prior]

        # motion_prior size : [batch, 3 * self.per_pix_k_size **2, H, W]
        return {'motion_prior': self.motion_prior_list, 'content_feature': content_feature}