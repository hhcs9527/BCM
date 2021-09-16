import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision

from DCNv2.dcn_v2 import DCN
from sub_modules.attention_zoo import eca_layer
from sub_modules.component_block import *
from pacnet.pac import PacConv2d, PacConvTranspose2d
from sub_modules.ConvLSTM_pytorch.convlstm import *

class Guide_Feature(nn.Module):

    def __init__(self, opt):
        super(Guide_Feature, self).__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel//2, stride_len = 1)

        # Downsample for patch level 1
        self.Downsample_Patch_lv1 = Guide_DCN_Block(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 2
        self.Downsample_Patch_lv2 = Guide_DCN_Block(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 3
        self.Downsample_Patch_lv3 = Guide_DCN_Block(in_c = opt.channel//2*4, out_c = opt.channel, stride_len = 2)

        # resolution is input/(4*4) here
        self.ASPP_conv = nn.Sequential(nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1), nn.LeakyReLU(0.2))

        # back to input resolution
        self.MASK_conv = nn.Sequential(nn.ConvTranspose2d(opt.channel*3, opt.channel, 3, stride=2, padding=1, output_padding = 1),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(opt.channel, opt.channel, 3, stride=2, padding=1, output_padding = 1),
                    nn.LeakyReLU(0.2), 
                    nn.Conv2d(opt.channel, 3, kernel_size = 1), 
                    nn.Sigmoid())

        
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
        
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 1)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 1)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 1)


        feature_lv2_1 = self.Patch_lv2_FE(images_lv2_1.contiguous())
        feature_lv2_2 = self.Patch_lv2_FE(images_lv2_2.contiguous())

        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)

        cat_feature = torch.cat((self.Downsample_Patch_lv1(self.Patch_lv1_FE(x)), 
                                self.Downsample_Patch_lv2(feature_lv2),
                                self.Downsample_Patch_lv3(feature_lv3)), 1)

        ASPP = self.ASPP_conv(cat_feature)
        blur_mask = self.MASK_conv(cat_feature)
        
        return x, feature_lv2, feature_lv3, ASPP, blur_mask


class Guide_Feature_Decoder(nn.Module):

    def __init__(self, opt):
        super(Guide_Feature_Decoder, self).__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel//2, stride_len = 1)

        
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
        
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 1)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 1)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 1)


        feature_lv2_1 = self.Patch_lv2_FE(images_lv2_1.contiguous())
        feature_lv2_2 = self.Patch_lv2_FE(images_lv2_2.contiguous())

        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
        feature_lv1 = self.Patch_lv1_FE(x)
        
        return feature_lv1, feature_lv2, feature_lv3



class Blur_Feature_Extractor(nn.Module):

    def __init__(self, opt):
        super().__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Downsample for patch level 1
        self.Downsample_Patch_lv1 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 2
        self.Downsample_Patch_lv2 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 3
        self.Downsample_Patch_lv3 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # resolution is input/(4*4) here
        attention = True
        if attention:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1), 
                nn.InstanceNorm2d(opt.channel*4), 
                eca_layer(channel = opt.channel*4, k_size = 3))
        else:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1),
                nn.InstanceNorm2d(opt.channel*4))

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

        cat_feature = torch.cat((self.Downsample_Patch_lv1(self.Patch_lv1_FE(x)), 
                                self.Downsample_Patch_lv2(feature_lv2),
                                self.Downsample_Patch_lv3(feature_lv3)), 1)

        ASPP = self.ASPP_conv(cat_feature)

        # out size : [batch, opt.channel*4, H/4, W/4]
        return ASPP


class Feature_Distribution(nn.Module):

    def __init__(self, opt):
        super().__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Downsample for patch level 1
        self.Downsample_Patch_lv1 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 2
        self.Downsample_Patch_lv2 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 3
        self.Downsample_Patch_lv3 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # resolution is input/(4*4) here
        attention = True
        if attention:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1), 
                nn.InstanceNorm2d(opt.channel*4), 
                eca_layer(channel = opt.channel*4, k_size = 3))
        else:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1),
                nn.InstanceNorm2d(opt.channel*4))
        
        # Mu of Blur distribution
        self.blur_mu = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        # Var for Blur distribution
        self.blur_var = get_DCN_norm_layer(opt.Norm, opt.channel*4)

        # Mapping Network
        self.Mapping_Network = nn.Sequential(
            get_DCN_norm_layer(opt.Norm, opt.channel*4),
            get_norm_layer(opt.Norm, opt.channel*4),
            get_norm_layer(opt.Norm, opt.channel*4),            
            get_DCN_norm_layer(opt.Norm, opt.channel*4),
            nn.InstanceNorm2d(opt.channel*4),
            eca_layer(channel = opt.channel*4, k_size = 3)
        )

        # Mu of Sharp distribution
        self.sharp_mu = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        # Var for Sharp distribution
        self.sharp_var = get_DCN_norm_layer(opt.Norm, opt.channel*4)


    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


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

        cat_feature = torch.cat((self.Downsample_Patch_lv1(self.Patch_lv1_FE(x)), 
                                self.Downsample_Patch_lv2(feature_lv2),
                                self.Downsample_Patch_lv3(feature_lv3)), 1)

        # Sample blur distribution
        blur_distribution = self.ASPP_conv(cat_feature)
        self.b_mu = self.blur_mu(blur_distribution)
        self.b_var = self.blur_var(blur_distribution)
        blur_feature = self.reparameterize(self.b_mu, self.b_var)

        # Sample sharp distribution
        sharp_distribution = self.Mapping_Network(blur_feature)
        self.s_mu = self.sharp_mu(sharp_distribution)
        self.s_var = self.sharp_var(sharp_distribution)
        sharp_feature = self.reparameterize(self.s_mu, self.s_var)

        # out size : [batch, opt.channel*4, H/4, W/4]
        return blur_feature, sharp_feature




'''
    channel * 2
'''
class Motion_Extractorg(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.space_to_depth = space_to_depth(4)
        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        self.non_linear_transform_lv1 = get_norm_layer('relu', opt.channel)
        self.non_linear_transform_lv2 = get_norm_layer('relu', opt.channel)

        self.feature_fusion_conv1x1 = nn.Sequential(\
                eca_layer(channel = opt.channel * 3, k_size = 3),
                nn.Conv2d(opt.channel * 3, opt.channel, kernel_size = 1))

        # resolution is input/(4*4) here
        #####
            # how to fusion these thress patch size information?
            # ASFF, ... keep looking
            # [x] pixshuff -> cat -> downsample
            # [ ] cat  -> pixshuff -> downsample
            # [ ] sum  -> pixshuff -> downsample
            # [x] multiple -> pixshuff -> downsample, overflow
        #####
        attention = True
        if attention:
            self.motion_fusion = nn.Sequential(\
                nn.Conv2d(opt.channel * 16, opt.channel * 2, kernel_size = 1),
                get_norm_layer('relu', opt.channel * 2),
                eca_layer(channel = opt.channel * 2, k_size = 3))
        else:
            self.motion_fusion = nn.Sequential(\
                nn.Conv2d(opt.channel * 16 * 3, opt.channel * 8, kernel_size = 3),
                nn.Conv2d(opt.channel * 8, opt.channel * 2, kernel_size = 1),
                get_norm_layer('relu', opt.channel * 2),nn.InstanceNorm2d(opt.channel * 2),
                nn.InstanceNorm2d(opt.channel*4))
    
    def residual_non_linear(self, feature):
        first_order = feature
        second_order = self.non_linear_transform_lv1(first_order) + first_order 
        return self.non_linear_transform_lv2(second_order) + first_order + second_order
        
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

        feature_lv1 = self.Patch_lv1_FE(x)

        feature_lv1, feature_lv2, feature_lv3 = self.residual_non_linear(feature_lv1), self.residual_non_linear(feature_lv2), self.residual_non_linear(feature_lv3)

        # with size (batch, opt.channel * 2 * 16, H/4, W/4)
        # mix them by multipling them 
        mix_feature = self.space_to_depth(self.feature_fusion_conv1x1(torch.cat((feature_lv1, feature_lv2, feature_lv3), 1)))

        # Sample motion distribution
        motion_distribution = self.motion_fusion(mix_feature)
        
        # out size : [batch, opt.channel * 2, H/4, W/4]
        return motion_distribution 


'''
    channel * 4
'''
class Motion_Extractor(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.space_to_depth = space_to_depth(4)
        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        self.non_linear_transform_lv1 = get_norm_layer('relu', opt.channel)
        self.non_linear_transform_lv2 = get_norm_layer('relu', opt.channel)

        self.feature_fusion_conv3x3 = nn.Sequential(
                nn.Conv2d(opt.channel * 3, opt.channel, kernel_size = 3, padding = 1))



        # resolution is input/(4*4) here
        #####
            # how to fusion these thress patch size information?
            # ASFF, ... keep looking
            # [x] pixshuff -> cat -> downsample
            # [x] cat  -> pixshuff -> downsample
            # [x] sum  -> pixshuff -> downsample
            # [x] multiple -> pixshuff -> downsample, overflow
        #####
        self.motion_lv1 = nn.Sequential(
                self.space_to_depth,
                nn.Conv2d(opt.channel * 16, opt.channel * 4, kernel_size = 1),
                get_norm_layer('relu', opt.channel * 4), nn.InstanceNorm2d(opt.channel * 4))
        
        self.motion_lv2 = nn.Sequential(
                self.space_to_depth,
                nn.Conv2d(opt.channel * 16, opt.channel * 4, kernel_size = 1),
                get_norm_layer('relu', opt.channel * 4), nn.InstanceNorm2d(opt.channel * 4))
        
        self.motion_lv3 = nn.Sequential(
                self.space_to_depth,
                nn.Conv2d(opt.channel * 16, opt.channel * 4, kernel_size = 1),
                get_norm_layer('relu', opt.channel * 4), nn.InstanceNorm2d(opt.channel * 4))

    
    def residual_non_linear(self, feature):
        first_order = feature
        second_order = self.non_linear_transform_lv1(first_order) + first_order 
        return self.non_linear_transform_lv2(second_order) + first_order + second_order
        
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

        feature_lv1 = self.Patch_lv1_FE(x)

        # with size (batch, opt.channel * 2 * 16, H/4, W/4)
        # mix them by multipling them 
        # Sample motion distribution
        motion = [self.motion_lv1(feature_lv1), self.motion_lv2(feature_lv2), self.motion_lv3(feature_lv3)]
        
        # out size : [batch, opt.channel * 2, H/4, W/4]
        return motion


class multi_patch_D(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.space_to_depth = space_to_depth(4)
        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        self.non_linear_transform_lv1 = get_norm_layer('relu', opt.channel)
        self.non_linear_transform_lv2 = get_norm_layer('relu', opt.channel)

        self.feature_fusion_conv1x1 = nn.Sequential(\
                eca_layer(channel = opt.channel * 3, k_size = 3),
                nn.Conv2d(opt.channel * 3, opt.channel, kernel_size = 1))

        #Conv2
        self.conv_lv2 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.non_linear_conv_lv2 = nn.Sequential(
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)
            )

        #Conv3
        self.conv_lv3 = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.non_linear_conv_lv3 = nn.Sequential(
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
            )

    def residual_non_linear(self, feature):
        first_order = feature
        second_order = self.non_linear_transform_lv1(first_order) + first_order 
        return self.non_linear_transform_lv2(second_order) + first_order + second_order
        
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

        feature_lv1 = self.Patch_lv1_FE(x)

        # mix them by 1*1 conv them
        mix_feature = self.feature_fusion_conv1x1(torch.cat((feature_lv1, feature_lv2, feature_lv3), 1))
        mix_feature = self.residual_non_linear(mix_feature)

        # Conv2 (follow the original encoder of DMPHN)
        # input size : (batch, opt.channel, H, W)
        x = self.conv_lv2(mix_feature)
        x = self.non_linear_conv_lv2(x) + x
        x = self.non_linear_conv_lv2(x) + x
        # Conv3
        # input size : (batch, opt.channel * 2, H/2, W/2)
        x = self.conv_lv3(x)
        x = self.non_linear_conv_lv3(x) + x
        discriminat_feature = self.non_linear_conv_lv3(x) + x
        
        # out size : [batch, opt.channel * 4, H/4, W/4]
        return discriminat_feature



class Motion(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        self.non_linear_transform_lv1 = get_norm_layer('relu', opt.channel)
        self.non_linear_transform_lv2 = get_norm_layer('relu', opt.channel)

        self.feature_fusion_conv3x3 = nn.Sequential(
                nn.Conv2d(opt.channel * 3, opt.channel*4, kernel_size = 3, padding = 1),
                nn.Conv2d(opt.channel * 4, opt.channel*4, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.Conv2d(opt.channel * 4, opt.channel*4, kernel_size = 3, padding = 1))


        # resolution is input/(4*4) here
        #####
            # how to fusion these thress patch size information?
            # ASFF, ... keep looking
            # [x] pixshuff -> cat -> downsample
            # [x] cat  -> pixshuff -> downsample
            # [x] sum  -> pixshuff -> downsample
            # [x] multiple -> pixshuff -> downsample, overflow
        #####

    
    def residual_non_linear(self, feature):
        first_order = feature
        second_order = self.non_linear_transform_lv1(first_order) + first_order 
        return self.non_linear_transform_lv2(second_order) + first_order + second_order
        
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

        feature_lv1 = self.Patch_lv1_FE(x)

        # with size (batch, opt.channel, H, W)
        feature_lv1, feature_lv2, feature_lv3 = self.residual_non_linear(feature_lv1), self.residual_non_linear(feature_lv2), self.residual_non_linear(feature_lv3)

        # with size (batch, opt.channel * 2 * 16, H/4, W/4)
        # mix them by multipling them 
        mix_feature = self.feature_fusion_conv3x3(torch.cat((feature_lv1, feature_lv2, feature_lv3), 1))
        
        # out size : [batch, opt.channel * 2, H/4, W/4]
        self.feature_list = [(mix_feature)]
        return (mix_feature)

class Motion_Atrous(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        self.non_linear_transform_lv1 = get_norm_layer(opt.Norm, opt.channel)
        self.non_linear_transform_lv2 = get_norm_layer(opt.Norm, opt.channel)

        self.feature_fusion_conv3x3 = nn.Sequential(
                nn.Conv2d(opt.channel * 3, opt.channel*4, kernel_size = 3, padding = 1),
                nn.Conv2d(opt.channel * 4, opt.channel*4, kernel_size = 3, padding = 4, dilation=4),
                nn.Conv2d(opt.channel * 4, opt.channel*4, kernel_size = 3, padding = 4, dilation=4),
                nn.LeakyReLU(0.2, True))


        # resolution is input/(4*4) here
        #####
            # how to fusion these thress patch size information?
            # ASFF, ... keep looking
            # [x] pixshuff -> cat -> downsample
            # [x] cat  -> pixshuff -> downsample
            # [x] sum  -> pixshuff -> downsample
            # [x] multiple -> pixshuff -> downsample, overflow
        #####

    
    def residual_non_linear(self, feature):
        first_order = feature
        second_order = self.non_linear_transform_lv1(first_order) + first_order 
        return self.non_linear_transform_lv2(second_order) + first_order + second_order
        
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

        feature_lv1 = self.Patch_lv1_FE(x)
        
        # with size (batch, opt.channel, H, W)
        feature_lv1, feature_lv2, feature_lv3 = self.residual_non_linear(feature_lv1), self.residual_non_linear(feature_lv2), self.residual_non_linear(feature_lv3)

        # with size (batch, opt.channel * 2 * 16, H/4, W/4)
        # mix them by multipling them 
        mix_feature = self.feature_fusion_conv3x3(torch.cat((feature_lv1, feature_lv2, feature_lv3), 1))
        self.feature_list = [mix_feature]
        
        # out size : [batch, opt.channel * 2, H/4, W/4]
        return mix_feature

class Motion_consecutive(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel]*3

        for i in range(self.patch_level, 0, -1):
            channel = channel_list[i-1]
            level = f'level{i}'
            if i == self.patch_level:
                self.Patch_FE[level] = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1)
            else:
                self.Patch_FE[level] = Guide_DCN_Block(in_c = channel, out_c = channel, stride_len = 1)
            
            self.get_motion[level] = nn.Sequential(
                nn.Conv2d(opt.channel, opt.channel*2**(i-1), kernel_size = 3, padding = 1, stride = 2**(i-1), dilation=1),
                nn.LeakyReLU(0.2, True))
    
        
    def get_image_level(self,x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        
    def forward(self, x):
        self.get_image_level(x)
        H = x.size(2)
        W = x.size(3)
        self.feature_list = []
        consecutive_list = []
        for i in range(self.patch_level, 0, -1):
            level = f'level{i}'
            if i == 1:
                feature_lv1 = self.Patch_FE[level](consecutive_list[3-i-1])
                consecutive_list.append(feature_lv1)

            elif i == 2:
                feature_lv2_1 = consecutive_list[3-i-1][:,:,0:int(H/2),:]
                feature_lv2_2 = consecutive_list[3-i-1][:,:,int(H/2):H,:]
                feature_lv2_1 = self.Patch_FE[level](feature_lv2_1.contiguous())
                feature_lv2_2 = self.Patch_FE[level](feature_lv2_2.contiguous())
                feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
                consecutive_list.append(feature_lv2)

            elif i == self.patch_level:
                feature_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
                feature_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
                feature_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
                feature_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
                feature_lv3_1 = self.Patch_FE[level](feature_lv3_1.contiguous())
                feature_lv3_2 = self.Patch_FE[level](feature_lv3_2.contiguous())
                feature_lv3_3 = self.Patch_FE[level](feature_lv3_3.contiguous())
                feature_lv3_4 = self.Patch_FE[level](feature_lv3_4.contiguous())
                feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                consecutive_list.append(feature_lv3)
 
        for i in range(3,0,-1):
            level = f'level{i}'
            self.feature_list.append(self.get_motion[level](feature_lv1))
            
        #for i in range(1, len(feature_list)):
        #    result_feature += feature_list[i]
        
        # out size : [batch, opt.channel * 4, H, W]
        return self.feature_list


class Motion_Patch(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel]*3

        for i in range(self.patch_level, 0, -1):
            channel = channel_list[i-1]
            level = f'level{i}'
            self.Patch_FE[level] = nn.Sequential(Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 2),
                                    nn.Conv2d(opt.channel, opt.channel*2**(i-1), kernel_size = 3, padding = 1, stride = 2**(i-1), dilation=1),
                                    nn.LeakyReLU(0.2, True))
        
    
    def get_image_level(self,x):
        H = x.size(2)
        W = x.size(3)
        self.images_lv1 = x
        self.images_lv2_1 = self.images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = self.images_lv1[:,:,int(H/2):H,:]
        
    def forward(self, x):
        self.get_image_level(x)
        H = x.size(2)
        W = x.size(3)
        self.feature_list = []
        for i in range(1, self.patch_level + 1):
            level = f'level{i}'
            if i == 1:
                feature_lv1 = self.Patch_FE[level](x)
                self.feature_list.append(feature_lv1)

            elif i == 2:
                feature_lv2_1 = self.images_lv1[:,:,0:int(H/2),:]
                feature_lv2_2 = self.images_lv1[:,:,int(H/2):H,:]
                feature_lv2_1 = self.Patch_FE[level](feature_lv2_1.contiguous())
                feature_lv2_2 = self.Patch_FE[level](feature_lv2_2.contiguous())
                feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
                self.feature_list.append(feature_lv2)

            elif i == self.patch_level:
                feature_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
                feature_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
                feature_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
                feature_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
                feature_lv3_1 = self.Patch_FE[level](feature_lv3_1.contiguous())
                feature_lv3_2 = self.Patch_FE[level](feature_lv3_2.contiguous())
                feature_lv3_3 = self.Patch_FE[level](feature_lv3_3.contiguous())
                feature_lv3_4 = self.Patch_FE[level](feature_lv3_4.contiguous())
                feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                self.feature_list.append(feature_lv3)

        # out size : [batch, opt.channel * 4, H, W]
        return self.feature_list #self.get_motion(feature_list[0]) 


class Motion_Patch_encoder(nn.Module):
    '''
        each patch with different network.
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''
    # must 連續, 考慮加入ASPP block

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.Res_ASPP = nn.ModuleDict()

        self.patch_level = 3
        # 32 -> 64 -> 128
        channel_list = [opt.channel * 2**(3-i) for i in range(1,4)]

        for i in range(self.patch_level, 0, -1):
            channel = channel_list[i-1]
            level = f'level{i}'
            if i == self.patch_level:
                self.Patch_FE[level] = nn.Sequential( 
                                        Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5),
                                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        get_norm_layer('leaky', channel)
                                        )
            else:
                self.Patch_FE[level] = self.Patch_FE[level] = nn.Sequential(                     
                                        Guide_DCN_Block(in_c = channel_list[i], out_c = channel, stride_len = 1, kernel_size = 3),
                                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        get_norm_layer('leaky', channel)
                                        )
            if i == 3 :
                self.get_motion[level] = nn.Sequential(               
                    nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                    nn.LeakyReLU(0.2, True))
            else:
                self.get_motion[level] = nn.Sequential(                                    
                    nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                    nn.LeakyReLU(0.2, True))
            #self.Res_ASPP[level] = RES_ASPP_Block(in_c = 3, out_c = channel, stride_len = 1)
    
        
    def get_image_level(self, x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        
    def forward(self, x):
        self.get_image_level(x)
        H = x.size(2)
        W = x.size(3)
        self.feature_list = []
        consecutive_list = []
        feature_dict = {}

        for i in range(self.patch_level, 0, -1):
            level = f'level{i}'
            if i == 1:
                feature_lv1 = self.Patch_FE[level](self.feature_list[3-i-1])
                self.feature_list.append(nn.Sigmoid()(self.get_motion[level](feature_lv1)))

            elif i == 2:
                feature_lv2_1 = self.feature_list[3-i-1][:,:,0:int(H/2),:]
                feature_lv2_2 = self.feature_list[3-i-1][:,:,int(H/2):H,:]
                feature_lv2_1 = self.Patch_FE[level](feature_lv2_1.contiguous())
                feature_lv2_2 = self.Patch_FE[level](feature_lv2_2.contiguous())
                feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
                self.feature_list.append(self.get_motion[level](feature_lv2))

            elif i == self.patch_level:
                feature_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
                feature_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
                feature_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
                feature_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
                feature_lv3_1 = self.Patch_FE[level](feature_lv3_1.contiguous())
                feature_lv3_2 = self.Patch_FE[level](feature_lv3_2.contiguous())
                feature_lv3_3 = self.Patch_FE[level](feature_lv3_3.contiguous())
                feature_lv3_4 = self.Patch_FE[level](feature_lv3_4.contiguous())
                feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                self.feature_list.append(self.get_motion[level](feature_lv3))


            
        for i in range(len(self.feature_list)-1, -1, -1):
            patch = f'patch_{2**(len(self.feature_list)-1-i)}'
            feature_dict[patch] = self.feature_list[i]
        
        # out size : [batch, opt.channel * 4, H, W]
        # self.feature_list -> dict[patch4, patch2, patch1]
        return feature_dict

class Motion_Patch_decoder(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''
    # patch 1 : [b, opt.channel * 4, H/4, W/4]  
    # patch 2 : [b, opt.channel * 2, H/2, W/2]  
    # patch 4 : [b, opt.channel * 1, H, W]                                         

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.PAC = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel * 2**(3-i) for i in range(1,4)]

        for i in range(1, self.patch_level+1):
            channel = channel_list[i-1]
            level = f'level{i}'

            if i != 3:
                self.Patch_FE[level] = nn.Sequential(
                                    Guide_DCN_Block(in_c = channel_list[i-1], out_c = channel_list[i], stride_len = 1, kernel_size = 3),
                                    nn.Conv2d(channel_list[i], channel_list[i], kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    #RES_ASPP_Block(in_c = channel_list[i], out_c = channel_list[i], stride_len = 1),
                                    get_norm_layer('leaky', channel_list[i]),
                                    nn.LeakyReLU(0.2, True))
            else:
                self.Patch_FE[level] = nn.Sequential(
                                    Guide_DCN_Block(in_c = channel_list[i-1], out_c = channel_list[i-1], stride_len = 1, kernel_size = 5),
                                    nn.Conv2d(channel_list[i-1], channel_list[i-1], kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                    #RES_ASPP_Block(in_c = 3, out_c = 3, stride_len = 1),
                                    get_norm_layer('leaky', channel_list[i-1]),
                                    nn.LeakyReLU(0.2, True))
            
            self.PAC[level] = PacConv2d(channel_list[i-1], channel_list[i-1], kernel_size=3, padding=1)
        
    
    def get_image_level(self, level, motion, input):
        patch = f'patch_{2**(level-1)}'
        if level == 1:
            return motion[patch]
        
        elif level == 2:
            self.images_lv1 = motion[patch] + input
            self.H = self.images_lv1.size(2)
            self.W = self.images_lv1.size(3)
        
        elif level == 3:
            x = motion[patch] + input
            self.H = x.size(2)
            self.W = x.size(3)
            self.images_lv2_1 = x[:,:,0:int(self.H/2),:]
            self.images_lv2_2 = x[:,:,int(self.H/2):self.H,:]
        

    def forward(self, x):
        # x dict -> [patch4, patch2, patch1]

        self.feature_list = []
        for i in range(1, self.patch_level + 1):
            level = f'level{i}'
            patch = f'patch_{2**(i-1)}'

            if i == 1:
                image = self.get_image_level(i, x, x[patch])
                feature_lv1 = self.Patch_FE[level](image)
                self.feature_list.append(feature_lv1)

            elif i == 2:
                self.get_image_level(i, x, self.feature_list[i-2])
                feature_lv2_1 = self.images_lv1[:,:,0:int(self.H/2),:]
                feature_lv2_2 = self.images_lv1[:,:,int(self.H/2):self.H,:]
                feature_lv2_1 = self.PAC[level](feature_lv2_1, x[patch][:,:,0:int(self.H/2),:])
                feature_lv2_2 = self.PAC[level](feature_lv2_2, x[patch][:,:,int(self.H/2):self.H,:])
                feature_lv2_1 = self.Patch_FE[level](feature_lv2_1.contiguous())
                feature_lv2_2 = self.Patch_FE[level](feature_lv2_2.contiguous())
                feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
                self.feature_list.append(feature_lv2)

            elif i == self.patch_level:
                self.get_image_level(i, x, self.feature_list[i-2])
                feature_lv3_1 = self.images_lv2_1[:,:,:,0:int(self.W/2)]
                feature_lv3_2 = self.images_lv2_1[:,:,:,int(self.W/2):self.W]
                feature_lv3_3 = self.images_lv2_2[:,:,:,0:int(self.W/2)]
                feature_lv3_4 = self.images_lv2_2[:,:,:,int(self.W/2):self.W]

                feature_lv3_1 = self.PAC[level](feature_lv3_1, x[patch][:,:,0:int(self.H/2),:][:,:,:,0:int(self.W/2)])
                feature_lv3_2 = self.PAC[level](feature_lv3_2, x[patch][:,:,0:int(self.H/2),:][:,:,:,int(self.W/2):self.W])
                feature_lv3_3 = self.PAC[level](feature_lv3_3, x[patch][:,:,int(self.H/2):self.H,:][:,:,:,0:int(self.W/2)])
                feature_lv3_4 = self.PAC[level](feature_lv3_4, x[patch][:,:,int(self.H/2):self.H,:][:,:,:,int(self.W/2):self.W])

                feature_lv3_1 = self.Patch_FE[level](feature_lv3_1.contiguous())
                feature_lv3_2 = self.Patch_FE[level](feature_lv3_2.contiguous())
                feature_lv3_3 = self.Patch_FE[level](feature_lv3_3.contiguous())
                feature_lv3_4 = self.Patch_FE[level](feature_lv3_4.contiguous())
                feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                self.feature_list.append(feature_lv3)

        # out size : [batch, opt.channel * 4, H, W]
        return self.feature_list[-1] #self.get_motion(feature_list[0]) 


class Motion_Patch_encoder_var(nn.Module):
    '''
        each patch with different network.
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''
    # must 連續, 考慮加入ASPP block

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.Conv = nn.ModuleDict()
        self.Fuse = nn.ModuleDict()

        self.patch_level = 3
        # 32 -> 64 -> 128
        channel_list = [opt.channel * 2**(3-i) for i in range(1,4)]

        for i in range(self.patch_level, 0, -1):
            channel = channel_list[i-1]
            level = f'level{i}'
            num_patch = 2 ** (i-1)
            if i == self.patch_level:
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.Patch_FE[patch_level] = nn.Sequential( 
                                        Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5),
                                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        get_norm_layer('leaky', channel)
                                        )
            #    self.Fuse[level] = nn.Sequential(
            #                Guide_DCN_Block(in_c = channel, out_c = channel, stride_len = 1, kernel_size = 5))
                
            else:
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.Patch_FE[patch_level] = self.Patch_FE[patch_level] = nn.Sequential(                     
                                        Guide_DCN_Block(in_c = channel_list[i], out_c = channel, stride_len = 1, kernel_size = 3),
                                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        get_norm_layer('leaky', channel)
                                        )
            #    self.Fuse[level] = nn.Sequential(
            #                Guide_DCN_Block(in_c = channel, out_c = channel, stride_len = 1))

            if i == 3 :
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.get_motion[patch_level] = nn.Sequential(               
                            nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                            nn.LeakyReLU(0.2, True))
                
            else:
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.get_motion[patch_level] = nn.Sequential(                                    
                            nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                            nn.LeakyReLU(0.2, True))
            
            #self.Conv[level] = nn.Sequential(
            #                get_norm_layer('leaky', channel))
            
            

            #self.Res_ASPP[level] = RES_ASPP_Block(in_c = 3, out_c = channel, stride_len = 1)
    
    def get_image_level(self, x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        
    def forward(self, x):
        self.get_image_level(x)
        H = x.size(2)
        W = x.size(3)
        self.feature_list = []
        consecutive_list = []
        feature_dict = {}

        for i in range(self.patch_level, 0, -1):
            level = f'level{i}'
            num_patch = 2 ** (i-1)
            if i == 1:
                feature_lv1 = [self.feature_list[3-i-1].contiguous()]
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv1 = self.Patch_FE[patch_level](feature_lv1[j])
                    #f_lv1 = self.Conv[level](self.get_motion[patch_level](f_lv1))
                #self.feature_list.append(self.Fuse[level](f_lv1.contiguous()))
                    f_lv1 = self.get_motion[patch_level](f_lv1)
                feature_dict[level] = [f_lv1]
                self.feature_list.append(f_lv1.contiguous())

            elif i == 2:
                feature_lv2 = [self.feature_list[3-i-1][:,:,0:int(H/2),:].contiguous(), self.feature_list[3-i-1][:,:,int(H/2):H,:].contiguous()]
                f_list = []
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv2 = self.Patch_FE[patch_level](feature_lv2[j])
                    f_list.append(self.get_motion[patch_level](f_lv2))
                f_lv2 = torch.cat((f_list[0], f_list[1]), 2)
                feature_dict[level] = f_list
                self.feature_list.append(f_lv2.contiguous())

            elif i == self.patch_level:
                feature_lv3 = [self.images_lv2_1[:,:,:,0:int(W/2)].contiguous(), self.images_lv2_1[:,:,:,int(W/2):W].contiguous(), self.images_lv2_2[:,:,:,0:int(W/2)].contiguous(), self.images_lv2_2[:,:,:,int(W/2):W].contiguous()]
                f_list = []
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv3 = self.Patch_FE[patch_level](feature_lv3[j])
                    f_list.append(self.get_motion[patch_level](f_lv3))
                feature_lv3_top = torch.cat((f_list[0], f_list[1]), 3)
                feature_lv3_bot = torch.cat((f_list[2], f_list[3]), 3)
                f_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                feature_dict[level] = f_list
                self.feature_list.append(f_lv3.contiguous())

        for i in range(len(self.feature_list)-1, -1, -1):
            patch = f'patch_{2**(len(self.feature_list)-1-i)}'
            feature_dict[patch] = self.feature_list[i]
        
        # out size : [batch, opt.channel * 4, H, W]
        # self.feature_list -> dict[patch4, patch2, patch1]
        return self.feature_list[-1]#feature_dict

class Motion_Patch_decoder_var(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.Conv = nn.ModuleDict()
        self.PAC = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel * 2**(3-i) for i in range(1,4)]

        for i in range(1, self.patch_level+1):
            channel = channel_list[i-1]
            level = f'level{i}'
            num_patch = 2 ** (i-1)

            if i != 3:
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.Patch_FE[patch_level] = nn.Sequential(
                                    Guide_DCN_Block(in_c = channel_list[i-1], out_c = channel_list[i], stride_len = 1, kernel_size = 3),
                                    nn.Conv2d(channel_list[i], channel_list[i], kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    #RES_ASPP_Block(in_c = channel_list[i], out_c = channel_list[i], stride_len = 1),
                                    get_norm_layer('leaky', channel_list[i]),
                                    nn.LeakyReLU(0.2, True))
                    self.PAC[patch_level] = PacConv2d(channel_list[i-1], channel_list[i-1], kernel_size=3, padding=1)
                #self.Conv[level] = nn.Sequential(
                #    get_norm_layer('leaky', channel_list[i]))
                #self.Fuse[level] = nn.Sequential(
                #    Guide_DCN_Block(in_c = channel_list[i], out_c = channel_list[i], stride_len = 1))

            else:
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.Patch_FE[patch_level] = nn.Sequential(
                                    Guide_DCN_Block(in_c = channel_list[i-1], out_c = 3, stride_len = 1, kernel_size = 5),
                                    nn.Conv2d(3, 3, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                    #RES_ASPP_Block(in_c = 3, out_c = 3, stride_len = 1),
                                    get_norm_layer('leaky', 3),
                                    nn.LeakyReLU(0.2, True))
                    self.PAC[patch_level] = PacConv2d(channel_list[i-1], channel_list[i-1], kernel_size=3, padding=1)
                #self.Conv[level] = nn.Sequential(
                #    get_norm_layer('leaky', 3))
                #self.Fuse[level] = nn.Sequential(
                #    Guide_DCN_Block(in_c = 3, out_c = 3, stride_len = 1, kernel_size = 5))
    

    def get_image_level(self, level, motion, input):
        patch = f'patch_{2**(level-1)}'
        if level == 1:
            return motion[patch]
        
        elif level == 2:
            self.images_lv1 = motion[patch] + input
            self.H = self.images_lv1.size(2)
            self.W = self.images_lv1.size(3)
        
        elif level == 3:
            x = motion[patch] + input
            self.H = x.size(2)
            self.W = x.size(3)
            self.images_lv2_1 = x[:,:,0:int(self.H/2),:]
            self.images_lv2_2 = x[:,:,int(self.H/2):self.H,:]
        

    def forward(self, x):
        # x dict -> [patch4, patch2, patch1]

        self.feature_list = []
        for i in range(1, self.patch_level + 1):
            level = f'level{i}'
            patch = f'patch_{2**(i-1)}'
            num_patch = 2 ** (i-1)
            if i == 1:
                feature_lv1 = [self.get_image_level(i, x, x[patch])]
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv1 = self.PAC[patch_level](feature_lv1[j], x[level][j])
                    f_lv1 = self.Patch_FE[patch_level](f_lv1)
                self.feature_list.append(f_lv1)

            elif i == 2:
                self.get_image_level(i, x, self.feature_list[i-2])
                feature_lv2 = [self.images_lv1[:,:,0:int(self.H/2),:].contiguous(), self.images_lv1[:,:,int(self.H/2):self.H,:].contiguous()]
                f_list = []
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv2 = self.PAC[patch_level](feature_lv2[j], x[level][j])
                    f_lv2 = self.Patch_FE[patch_level](f_lv2)
                    f_list.append(f_lv2)
                feature_lv2 = torch.cat((f_list[0], f_list[1]), 2)
                self.feature_list.append(feature_lv2.contiguous())

            elif i == self.patch_level:
                self.get_image_level(i, x, self.feature_list[i-2])
                feature_lv3 = [self.images_lv2_1[:,:,:,0:int(self.W/2)].contiguous(), self.images_lv2_1[:,:,:,int(self.W/2):self.W].contiguous(), self.images_lv2_2[:,:,:,0:int(self.W/2)].contiguous(), self.images_lv2_2[:,:,:,int(self.W/2):self.W].contiguous()]
                f_list = []
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv3 = self.PAC[patch_level](feature_lv3[j], x[level][j])
                    f_lv3 = self.Patch_FE[patch_level](f_lv3)
                    f_list.append(f_lv3)
                feature_lv3_top = torch.cat((f_list[0], f_list[1]), 3)
                feature_lv3_bot = torch.cat((f_list[2], f_list[3]), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                self.feature_list.append(feature_lv3.contiguous())
                #self.feature_list.append(self.Fuse[level](feature_lv3))           
            
        # out size : [batch, opt.channel * 4, H, W]
        return self.feature_list[-1] #self.get_motion(feature_list[0]) 



class Motion_Patch_var(nn.Module):
    '''
        each patch with different network.
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''
    # must 連續, 考慮加入ASPP block

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.Conv = nn.ModuleDict()
        self.Fuse = nn.ModuleDict()

        self.patch_level = 3
        # 32 -> 64 -> 128
        channel_list = [opt.channel * 2**(3-i) for i in range(1,4)]

        for i in range(self.patch_level, 2, -1):
            channel = 32 #channel_list[i-1]
            level = f'level{i}'
            num_patch = 2 ** (i-1)
            if i == self.patch_level:
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.Patch_FE[patch_level] = nn.Sequential( 
                                        Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 4, dilation=1),
                                        RES_FULL_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        get_norm_layer('leaky', channel)
                                        )
            #    self.Fuse[level] = nn.Sequential(
            #                Guide_DCN_Block(in_c = channel, out_c = channel, stride_len = 1, kernel_size = 5))

            if i == 3 :
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    self.get_motion[patch_level] = nn.Sequential(               
                            nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                            nn.LeakyReLU(0.2, True))
            
            #self.Conv[level] = nn.Sequential(
            #                get_norm_layer('leaky', channel))
            
            

            #self.Res_ASPP[level] = RES_ASPP_Block(in_c = 3, out_c = channel, stride_len = 1)
    
    def get_image_level(self, x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        
    def forward(self, x):
        self.get_image_level(x)
        H = x.size(2)
        W = x.size(3)
        self.feature_list = []
        consecutive_list = []
        feature_dict = {}

        for i in range(self.patch_level, 2, -1):
            level = f'level{i}'
            num_patch = 2 ** (i-1)
            if i == 1:
                feature_lv1 = [self.feature_list[3-i-1].contiguous()]
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv1 = self.Patch_FE[patch_level](feature_lv1[j])
                    #f_lv1 = self.Conv[level](self.get_motion[patch_level](f_lv1))
                #self.feature_list.append(self.Fuse[level](f_lv1.contiguous()))
                    f_lv1 = self.get_motion[patch_level](f_lv1)
                feature_dict[level] = [f_lv1]
                self.feature_list.append(f_lv1.contiguous())

            elif i == 2:
                feature_lv2 = [self.feature_list[3-i-1][:,:,0:int(H/2),:].contiguous(), self.feature_list[3-i-1][:,:,int(H/2):H,:].contiguous()]
                f_list = []
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv2 = self.Patch_FE[patch_level](feature_lv2[j])
                    f_list.append(self.get_motion[patch_level](f_lv2))
                f_lv2 = torch.cat((f_list[0], f_list[1]), 2)
                feature_dict[level] = f_list
                self.feature_list.append(f_lv2.contiguous())

            elif i == self.patch_level:
                feature_lv3 = [self.images_lv2_1[:,:,:,0:int(W/2)].contiguous(), self.images_lv2_1[:,:,:,int(W/2):W].contiguous(), self.images_lv2_2[:,:,:,0:int(W/2)].contiguous(), self.images_lv2_2[:,:,:,int(W/2):W].contiguous()]
                f_list = []
                for j in range(num_patch):
                    patch_level = f'{level}_{j}'
                    f_lv3 = self.Patch_FE[patch_level](feature_lv3[j])
                    f_list.append(self.get_motion[patch_level](f_lv3))
                feature_lv3_top = torch.cat((f_list[0], f_list[1]), 3)
                feature_lv3_bot = torch.cat((f_list[2], f_list[3]), 3)
                f_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                feature_dict[level] = f_list
                self.feature_list.append(f_lv3.contiguous())

        #for i in range(len(self.feature_list)-1, -1, -1):
        #    patch = f'patch_{2**(len(self.feature_list)-1-i)}'
        #    feature_dict[patch] = self.feature_list[i]
        
        # out size : [batch, opt.channel * 4, H, W]
        # self.feature_list -> dict[patch4, patch2, patch1]
        return self.feature_list[-1]#feature_dict








class Motion_DCN_LSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Downsample for patch level 1
        self.Downsample_Patch_lv1 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 2
        self.Downsample_Patch_lv2 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 3
        self.Downsample_Patch_lv3 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # resolution is input/(4*4) here
        attention = True
        if attention:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1)) 
                #nn.InstanceNorm2d(opt.channel*4), 
                #eca_layer(channel = opt.channel*4, k_size = 3))
        else:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1),
                nn.InstanceNorm2d(opt.channel*4))

        self.recurrent = opt.Recurrent_times - 2 
        self.ConvLSTM = nn.ModuleDict()


        for i in range(self.recurrent):
            level = f'level{i}'
            #self.ConvLSTM[level] = get_norm_layer('leaky', opt.channel*4)
            self.ConvLSTM[level] = ConvLSTM(input_dim = opt.channel*4, hidden_dim=opt.channel*4, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)



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

        cat_feature = torch.cat((self.Downsample_Patch_lv1(self.Patch_lv1_FE(x)), 
                                self.Downsample_Patch_lv2(feature_lv2),
                                self.Downsample_Patch_lv3(feature_lv3)), 1)

        ASPP = self.ASPP_conv(cat_feature)
        return ASPP


class Motion_DCN(nn.Module):

    def __init__(self, opt):
        super().__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1, kernel_size = 5)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1, kernel_size = 5)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1, kernel_size = 5)

        # resolution is input/(4*4) here
        attention = True
        if attention:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, 3*3*3, kernel_size = 3)) 
                #nn.InstanceNorm2d(opt.channel*4), 
                #eca_layer(channel = opt.channel*4, k_size = 3))
        else:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1),
                nn.InstanceNorm2d(opt.channel*4))

        self.recurrent = opt.Recurrent_times - 2 
        self.ConvLSTM = nn.ModuleDict()


        #for i in range(self.recurrent):
        #    level = f'level{i}'
        #    #self.ConvLSTM[level] = get_norm_layer('leaky', opt.channel*4)
        #    self.ConvLSTM[level] = ConvLSTM(input_dim = opt.channel*4, hidden_dim=opt.channel*4, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)



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

        cat_feature = torch.cat((self.Patch_lv1_FE(x), feature_lv2, feature_lv3), 1)

        #ASPP = self.ASPP_conv(cat_feature)
        return cat_feature #ASPP






'''


class Motion_DCN_LSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = opt.channel, stride_len = 1)

        # Downsample for patch level 1
        self.Downsample_Patch_lv1 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 2
        self.Downsample_Patch_lv2 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # Downsample for patch level 3
        self.Downsample_Patch_lv3 = Guide_Block_FPN_downsample(in_c = opt.channel, out_c = opt.channel, stride_len = 4)

        # resolution is input/(4*4) here
        attention = True
        if attention:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1)) 
                #nn.InstanceNorm2d(opt.channel*4), 
                #eca_layer(channel = opt.channel*4, k_size = 3))
        else:
            self.ASPP_conv = nn.Sequential(\
                nn.Conv2d(opt.channel*3, opt.channel*4, kernel_size = 1),
                nn.InstanceNorm2d(opt.channel*4))

        self.recurrent = opt.Recurrent_times - 2 
        self.ConvLSTM = nn.ModuleDict()

        for i in range(self.recurrent):
            level = f'level{i}'
            self.ConvLSTM[level] = get_norm_layer('leaky', opt.channel*4)
            #ConvLSTM(input_dim = opt.channel*4, hidden_dim=opt.channel*4, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)



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

        cat_feature = torch.cat((self.Downsample_Patch_lv1(self.Patch_lv1_FE(x)), 
                                self.Downsample_Patch_lv2(feature_lv2),
                                self.Downsample_Patch_lv3(feature_lv3)), 1)

        ASPP = self.ASPP_conv(cat_feature)
        return ASPP
        # out size : [batch, opt.channel*4, H/4, W/4]
        recurrent_motion = [ASPP]
        hidden_state = None
        for i in range(self.recurrent):
            level = f'level{i}'
            #motion = recurrent_motion[i]
            #layer_outputs, hidden_state = self.ConvLSTM[level](torch.unsqueeze(recurrent_motion [i], 1), hidden_state)
            #recurrent_motion .append(torch.squeeze(layer_outputs[0], 1))
            recurrent_motion.append(self.ConvLSTM[level](recurrent_motion[i]))

        # out size : [batch, opt.channel * 4 * (opt.recurrent - 1) , H, W] 
        # self.feature_list -> dict[patch4, patch2, patch1]

        return torch.cat(recurrent_motion, dim = 1)#feature_dict
'''

class Sharp_Distribution(nn.Module):

    def __init__(self, opt):
        super().__init__()

        channel = opt.channel
        self.k_size = 3
        self.recurrent = opt.Recurrent_times - 2
        self.ConvLSTM = nn.ModuleDict()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Mapping Network
        self.Mapping_Network_DCN = nn.Sequential(
                            DCN(channel*3, channel*3, kernel_size=(3, 3), stride=1, padding = 1),
                            nn.Conv2d(channel*3, self.k_size ** 3, kernel_size = 3, stride = 1, padding = 1))

        for i in range(self.recurrent):
            level = f'level{i}'
            self.ConvLSTM[level] = ConvLSTM(input_dim = self.k_size**3, hidden_dim=self.k_size**3, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)

        # Down to H/4, W/4 prior motion and cat them
        self.Mapping_Network_motion_prior = nn.Sequential(
                            DCN(self.k_size**3 * (self.recurrent + 1), channel, kernel_size=(3, 3), stride=4, padding = 1),
                            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1))
       
        

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

        motion_ow = self.Mapping_Network_DCN(torch.cat((self.Patch_lv1_FE(x), feature_lv2, feature_lv3), 1))
        
        
        self.motion_feature_list = [motion_ow]

        hidden_state = None
        for i in range(self.recurrent):
            level = f'level{i}'
            motion = self.motion_feature_list[i]
            layer_outputs, hidden_state = self.ConvLSTM[level](torch.unsqueeze(self.motion_feature_list[i], 1), hidden_state)
            self.motion_feature_list.append(torch.squeeze(layer_outputs[0], 1))

        # get motion prior
        motion_prior = self.Mapping_Network_motion_prior(torch.cat(self.motion_feature_list, 1))

        self.feature_list = self.motion_feature_list

        # prior out size : [batch, opt.channel*4, H/4, W/4]
        # motion_ow : [batch, 27, H, W] -> for deformable conv
        return {'motion_ow' : self.motion_feature_list, 'motion_prior': motion_prior}


class Guide_pixel_offset(nn.Module):

    def __init__(self, opt):
        super().__init__()

        channel = opt.channel
        self.k_size = 3
        self.per_pix_k_size = 3
        self.recurrent = opt.Recurrent_times
        self.ConvLSTM = nn.ModuleDict()
        self.ConvLSTM_per_pix_w = nn.ModuleDict()

        # Patch level 1 feature extractor
        self.Patch_lv1_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 2 feature extractor
        self.Patch_lv2_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Patch level 3 feature extractor
        self.Patch_lv3_FE = Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5)

        # Mapping Network
        #self.Mapping_Network_DCN = nn.Sequential(
        #                    DCN(channel*3, channel*3, kernel_size=(3, 3), stride=1, padding = 1),
        #                    nn.Conv2d(channel*3, self.k_size ** 3, kernel_size = 3, stride = 1, padding = 1))

        #for i in range(self.recurrent):
        #    level = f'level{i}'
        #    self.ConvLSTM[level] = ConvLSTM(input_dim = self.k_size**3, hidden_dim=self.k_size**3, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)

        self.Mapping_Network_motion_prior = nn.Sequential(
                            #DCN(channel*3, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            nn.Conv2d(channel*3, channel, kernel_size=(3, 3), stride=1, padding = 1),
                            get_norm_layer(opt.Norm, channel),
                            nn.Conv2d(channel, 3 * self.per_pix_k_size ** 2 , kernel_size = 3 , stride = 1, padding = 1))       

        for i in range(self.recurrent - 1):
            level = f'level{i}'
            self.ConvLSTM[level] = ConvLSTM(input_dim = 3 * self.per_pix_k_size ** 2, hidden_dim=3 * self.per_pix_k_size ** 2, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)

        

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

        #motion_ow = self.Mapping_Network_DCN(torch.cat((self.Patch_lv1_FE(x), feature_lv2, feature_lv3), 1))

        #self.motion_feature_list = [motion_ow]

        # get motion prior
        motion_prior = self.Mapping_Network_motion_prior(torch.cat((self.Patch_lv1_FE(x), feature_lv2, feature_lv3), 1))
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