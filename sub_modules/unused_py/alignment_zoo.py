import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from DCNv2.DCN.dcn_v2 import DCN, DCNv2
from sub_modules.component_block import *

class PCD_alignment(nn.Module):
    '''
        reference code:
            https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py
        
        concept:
            1.  borrow the concept from EDVR. 
                we use the multi-scale of the offset here, try to maximize the offset of two latent feature.

            2.  we also borrow the concept from 
                Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network with Optical Flow Guided Training
                we use the conv to generate weight, and offset for deformable conv v2.
    '''
    def __init__(self, opt):
        super().__init__()
        '''
            # Pyramid has three levels:
            # L3: level 3, 1/4 spatial size
            # L2: level 2, 1/2 spatial size
            # L1: level 1, original spatial size
        '''
        self.offset_conv = nn.ModuleDict()
        self.offset_fusion = nn.ModuleDict()
        self.offset_transpose_conv = nn.ModuleDict()
        self.feat_transpose_conv = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.dcn = nn.ModuleDict()
        self.gen_ow = nn.ModuleDict()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.paramid_lv1 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size = 3, stride = 2, padding = 1)
        self.paramid_lv2 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size = 3, stride = 2, padding = 1)

        for i in range(3,0,-1):
            level = f'l{i}'
            self.offset_conv[level] = nn.Sequential(nn.Conv2d(opt.channel*4*2, opt.channel*4, kernel_size=1),
                                                    nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1))
            if i < 3:
                self.offset_fusion[level] = nn.Conv2d(3*3*2*2, 3*3*2, kernel_size=1)
            
            self.dcn[level] = DCNv2(in_channels = opt.channel*4, out_channels = opt.channel*4, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
            self.gen_ow[level] = nn.Conv2d(opt.channel*4, 3*3*3, kernel_size = 3, padding=1)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(opt.channel*4*2, opt.channel*4, kernel_size = 1)
            
            if i > 1:
                self.offset_transpose_conv[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(3*3*2, 3*3*2, kernel_size = 3, padding=1)) 

                self.feat_transpose_conv[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(in_channels = opt.channel*4, out_channels = opt.channel*4, kernel_size = 3, padding = 1))

    def get_paramid_feature(self, center_feature_lv1, to_align_feature_lv1):
        '''
            get the paramid feature for 3 levels
            # L3: level 3, batch, opt.channel*4, H/16, H/16
            # L2: level 2, batch, opt.channel*4, H/8, H/8
            # L1: level 1, batch, opt.channel*4, H/4, H/4
        '''
        center_feature_lv2, to_align_feature_lv2 = self.paramid_lv1(center_feature_lv1), self.paramid_lv1(to_align_feature_lv1)
        center_feature_lv3, to_align_feature_lv3 = self.paramid_lv2(center_feature_lv2), self.paramid_lv2(to_align_feature_lv2)

        return [center_feature_lv1, center_feature_lv2, center_feature_lv3], [to_align_feature_lv1, to_align_feature_lv2, to_align_feature_lv3]
    
    def get_offset_weight(self, feature, layer):
        '''
            feature with shape (batch, opt.channel*4, H/4, H/4)
            get the offset, weight for DCNv2
        '''
        ow = layer(feature)
        offset, weight = torch.split(ow, [3*3*2, 3*3], dim=1)
        return offset, weight
    
    def forward(self, center_feature_lv1, to_align_feature_lv1):
        '''
            input list:
                center_feature (list[Tensor]) : align standard
                to_align_feature (list[Tensor]) : feature try to align

            output:
                1. sum of offset in each level
                2. alignment feature, size : batch, opt.channel*4, H/4, H/4
        ''' 
        center_feature, to_align_feature = self.get_paramid_feature(center_feature_lv1, to_align_feature_lv1)
        upsampled_offset, upsampled_feat, total_offset = None, None, torch.tensor(0.0).cuda()

        for i in range(3,0,-1):
            level = f'l{i}'
            cat_feature = torch.cat([center_feature[i-1],to_align_feature[i-1]], 1)    
            cat_feature = self.lrelu(self.offset_conv[level](cat_feature))
            offset, weight = self.get_offset_weight(cat_feature, self.gen_ow[level])

            if i < 3:
                offset = self.offset_fusion[level](torch.cat([offset, upsampled_offset], dim=1))
            
            feat = self.dcn[level](cat_feature, offset, weight)
            
            if i < 3:
                feat = self.lrelu(self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1)))
            
            if i > 1:
                upsampled_offset = self.offset_transpose_conv[level](offset)
                upsampled_feat = self.feat_transpose_conv[level](feat)
        
            total_offset += torch.sum(offset)

        return feat, total_offset



# with channel*2 version
class FPN_alignment8(nn.Module):
    '''
        reference code:
            https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py
        
        concept:
            1.  borrow the concept from EDVR. 
                we use the multi-scale of the offset here, try to maximize the offset of two latent feature.

            2.  we also borrow the concept from 
                Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network with Optical Flow Guided Training
                we use the conv to generate weight, and offset for deformable conv v2.
    '''
    def __init__(self, opt):
        super().__init__()
        '''
            # Pyramid has three levels:
            # L3: level 3, 1/4 spatial size
            # L2: level 2, 1/2 spatial size
            # L1: level 1, original spatial size
        '''
        self.offset_conv = nn.ModuleDict()
        self.offset_fusion = nn.ModuleDict()
        self.offset_transpose_conv = nn.ModuleDict()
        self.feat_transpose_conv = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.dcn = nn.ModuleDict()
        self.gen_ow = nn.ModuleDict()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.paramid_lv1 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size = 3, stride = 2, padding = 1)
        self.paramid_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size = 3, stride = 2, padding = 1)

        for i in range(3,0,-1):
            level = f'l{i}'
            self.offset_conv[level] = nn.Sequential(nn.Conv2d(opt.channel*2*2, opt.channel*2, kernel_size=1),
                                                   # nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1))
                                                    nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1))
            if i < 3:
                #self.offset_fusion[level] = nn.Conv2d(3*3*2*2, 3*3*2, kernel_size=3, padding=1)
                self.offset_fusion[level] = nn.Conv2d(3*3*2*2, 3*3*2, kernel_size=1)
            
            self.dcn[level] = DCNv2(in_channels = opt.channel*2, out_channels = opt.channel*2, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
            self.gen_ow[level] = nn.Conv2d(opt.channel*2, 3*3*3, kernel_size = 3, padding=1)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(opt.channel*2*2, opt.channel*2, kernel_size = 1)
            
            if i > 1:
                self.offset_transpose_conv[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(3*3*2, 3*3*2, kernel_size = 3, padding=1)) 

                self.feat_transpose_conv[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(in_channels = opt.channel*2, out_channels = opt.channel*2, kernel_size = 3, padding = 1))
    
    def get_paramid_feature(self, center_feature_lv1, to_align_feature_lv1):
        '''
            get the paramid feature for 3 levels
            # L3: level 3, batch, opt.channel*2, H/16, H/16
            # L2: level 2, batch, opt.channel*2, H/8, H/8
            # L1: level 1, batch, opt.channel*2, H/4, H/4
        '''
        center_feature_lv2, to_align_feature_lv2 = self.paramid_lv1(center_feature_lv1), self.paramid_lv1(to_align_feature_lv1)
        center_feature_lv3, to_align_feature_lv3 = self.paramid_lv2(center_feature_lv2), self.paramid_lv2(to_align_feature_lv2)

        return [center_feature_lv1, center_feature_lv2, center_feature_lv3], [to_align_feature_lv1, to_align_feature_lv2, to_align_feature_lv3]
    
    def get_offset_weight(self, feature, layer):
        '''
            feature with shape (batch, opt.channel*2, H/4, H/4)
            get the offset, weight for DCNv2
        '''
        ow = layer(feature)
        offset, weight = torch.split(ow, [3*3*2, 3*3], dim=1)
        return offset, weight
    
    def forward(self, center_feature_lv1, to_align_feature_lv1):
        '''
            input list:
                center_feature (list[Tensor]) : align standard
                to_align_feature (list[Tensor]) : feature try to align

            output:
                1. sum of offset in each level
                2. alignment feature, size : batch, opt.channel*2, H/4, H/4
        ''' 
        center_feature, to_align_feature = self.get_paramid_feature(center_feature_lv1, to_align_feature_lv1)
        upsampled_offset, upsampled_feat, total_offset = None, None, torch.tensor(0.0).cuda()

        for i in range(3,0,-1):
            level = f'l{i}'
            cat_feature = torch.cat([center_feature[i-1],to_align_feature[i-1]], 1)    
            cat_feature = self.lrelu(self.offset_conv[level](cat_feature))
            offset, weight = self.get_offset_weight(cat_feature, self.gen_ow[level])

            if i < 3:
                offset = self.offset_fusion[level](torch.cat([offset, upsampled_offset], dim=1))
            
            feat = self.dcn[level](cat_feature, offset, weight)
            
            if i < 3:
                feat = self.lrelu(self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1)))
            
            if i > 1:
                upsampled_offset = self.offset_transpose_conv[level](offset)
                upsampled_feat = self.feat_transpose_conv[level](feat)
        
            total_offset += torch.mean(offset)

        return feat, total_offset


class IMG_alignment(nn.Module):
    '''
        reference code:
            https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py
        
        concept:
            1.  borrow the concept from EDVR. 
                we use the multi-scale of the offset here, try to maximize the offset of two latent feature.

            2.  we also borrow the concept from 
                Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network with Optical Flow Guided Training
                we use the conv to generate weight, and offset for deformable conv v2.
    '''
    def __init__(self, opt):
        super().__init__()
        '''
            # Pyramid has three levels:
            # L3: level 3, 1/4 spatial size
            # L2: level 2, 1/2 spatial size
            # L1: level 1, original spatial size
        '''
        self.offset_conv = nn.ModuleDict()
        self.offset_fusion = nn.ModuleDict()
        self.offset_transpose_conv = nn.ModuleDict()
        self.feat_transpose_conv = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.dcn = nn.ModuleDict()
        self.gen_ow = nn.ModuleDict()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.paramid_lv1 = nn.Conv2d(3, opt.channel, kernel_size = 3, stride = 2, padding = 1)
        self.paramid_lv2 = nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, stride = 2, padding = 1)

        for i in range(3,0,-1):
            level = f'l{i}'

            if i == 1:
                self.offset_conv[level] = nn.Sequential(nn.Conv2d(3*2, 3, kernel_size=1),
                                                    nn.Conv2d(3, opt.channel, kernel_size=3, padding=1))
                self.feat_conv[level] = nn.Conv2d(opt.channel*2, 3, kernel_size = 1)
            else:
                self.offset_conv[level] = nn.Sequential(nn.Conv2d(opt.channel*2, opt.channel, kernel_size=1),
                                                    nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1))

            if i < 3:
                self.offset_fusion[level] = nn.Conv2d(3*3*2*2, 3*3*2, kernel_size=1)

            self.gen_ow[level] = nn.Conv2d(opt.channel, 3*3*3, kernel_size = 3, padding=1)
            self.dcn[level] = DCNv2(in_channels = opt.channel, out_channels = opt.channel, kernel_size = 3, stride = 1, padding = 1, dilation = 1)

            if i == 2:
                self.feat_conv[level] = nn.Conv2d(opt.channel*2, opt.channel, kernel_size = 1)
            
            if i > 1:
                #self.dcn[level] = DCNv2(in_channels = opt.channel, out_channels = opt.channel, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
                self.offset_transpose_conv[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(3*3*2, 3*3*2, kernel_size = 3, padding=1)) 

                self.feat_transpose_conv[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(in_channels = opt.channel, out_channels = opt.channel, kernel_size = 3, padding = 1))


    def get_paramid_feature(self, center_feature_lv1, to_align_feature_lv1):
        '''
            get the paramid feature for 3 levels
            # L3: level 3, batch, 3, H/4, H/4
            # L2: level 2, batch, 3, H/2, H/2
            # L1: level 1, batch, 3, H, H
        '''
        center_feature_lv2, to_align_feature_lv2 = self.paramid_lv1(center_feature_lv1), self.paramid_lv1(to_align_feature_lv1)
        center_feature_lv3, to_align_feature_lv3 = self.paramid_lv2(center_feature_lv2), self.paramid_lv2(to_align_feature_lv2)

        return [center_feature_lv1, center_feature_lv2, center_feature_lv3], [to_align_feature_lv1, to_align_feature_lv2, to_align_feature_lv3]
    
    def get_offset_weight(self, feature, layer):
        '''
            feature with shape (batch, 3, H, W)
            get the offset, weight for DCNv2
        '''
        ow = layer(feature)
        offset, weight = torch.split(ow, [3*3*2, 3*3], dim=1)
        return offset, weight
    
    def forward(self, center_feature_lv1, to_align_feature_lv1):
        '''
            input list:
                center_feature (list[Tensor]) : align standard
                to_align_feature (list[Tensor]) : feature try to align

            output:
                1. sum of offset in each level
                2. alignment feature, size : batch, 3, H, W
        ''' 
        center_feature, to_align_feature = self.get_paramid_feature(center_feature_lv1, to_align_feature_lv1)
        upsampled_offset, upsampled_feat, total_offset = None, None, torch.tensor(0.0).cuda()

        for i in range(3,0,-1):
            level = f'l{i}'
            cat_feature = torch.cat([center_feature[i-1],to_align_feature[i-1]], 1)    
            cat_feature = self.lrelu(self.offset_conv[level](cat_feature))
            offset, weight = self.get_offset_weight(cat_feature, self.gen_ow[level])
            if i < 3:
                offset = self.offset_fusion[level](torch.cat([offset, upsampled_offset], dim=1))
            
            feat = self.dcn[level](cat_feature, offset, weight)
            
            if i < 3:
                feat = self.lrelu(self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1)))
            
            if i > 1:
                upsampled_offset = self.offset_transpose_conv[level](offset)
                upsampled_feat = self.feat_transpose_conv[level](feat)
        
            total_offset += torch.mean(offset)

        return feat, total_offset
