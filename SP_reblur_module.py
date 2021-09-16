import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_Res_Block(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(channel, channel, dilation = 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel, channel, dilation = 1, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.res_block(x)

class SP_module(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, in_channels, strip_size = 1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((strip_size, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, strip_size))

        self.conv_h = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False))
        self.conv_v = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False))
        self.conv_fusion = nn.Sequential(
                            Base_Res_Block(in_channels),
                            nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride=1),
                            )
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        _, _, h, w = x.size()
        x_horizon = F.interpolate(self.conv_h(self.pool_h(x)), (h, w), mode='bilinear', align_corners=True)
        x_vertical = F.interpolate(self.conv_v(self.pool_v(x)), (h, w), mode='bilinear', align_corners=True)
        sp_result = self.conv_fusion(x_horizon + x_vertical)
        return sp_result

class SP_SFT(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, in_channels, strip_size = 1):
        super().__init__()
        self.SP_scale = nn.Sequential(
                            SP_module(in_channels),
                            Base_Res_Block(in_channels),
                            )

        self.SP_shift = nn.Sequential(
                            SP_module(in_channels),
                            Base_Res_Block(in_channels),
                            )

        #self.instansnorm = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        scale = self.SP_scale(x)
        shift = self.SP_shift(x)

        return  x * scale + shift

class SP_reblur_attention(nn.Module):
    def __init__(self, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 5
        self.per_pix_k_size = 3
        self.reblur_attention = nn.Sequential(
                                #SP_module(input_channel),
                                SP_SFT(input_channel, strip_size = 1),
                                nn.Conv2d(input_channel, input_channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Base_Res_Block(input_channel),
                                nn.Conv2d(input_channel, input_channel, kernel_size = 1, stride=1),
                                nn.Sigmoid(),        
                                )  

        if self.ratio > 1:
            self.reblur_filer = nn.Sequential(
                                nn.Upsample(scale_factor=self.ratio, mode='bilinear', align_corners=True),
                                SP_SFT(input_channel, strip_size = 1),
                                Base_Res_Block(input_channel),
                                nn.Conv2d(input_channel, self.per_pix_k_size ** 2, kernel_size = 1, stride=1),
                                nn.LeakyReLU(0.2),
                                )
        else:
            self.reblur_filer = nn.Sequential(
                                SP_SFT(input_channel, strip_size = 1),
                                Base_Res_Block(input_channel),
                                nn.Conv2d(input_channel, self.per_pix_k_size ** 2, kernel_size = 1, stride=1),
                                nn.LeakyReLU(0.2),
                                )

    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        attention_feature = self.reblur_attention(feature) * feature

        reblur_filter = self.reblur_filer(feature - attention_feature)

        return {'reblur_filter' : reblur_filter, 'attention_feature' : attention_feature}

class blur_understanding_module(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self):
        super().__init__() 
        self.kernel = 3

    def unfold_and_permute(self, tensor, kernel, stride=1, pad=-1):
        if pad < 0:
            pad = (kernel - 1) // 2
        tensor = F.pad(tensor, (pad, pad, pad, pad))
        tensor = tensor.unfold(2, kernel, stride)
        tensor = tensor.unfold(3, kernel, stride)
        N, C, H, W, _, _ = tensor.size()
        tensor = tensor.reshape(N, C, H, W, -1)
        tensor = tensor.permute(0, 2, 3, 1, 4)
        return tensor

    def weight_permute_reshape(self, tensor, F, S2):
        N, C, H, W = tensor.size()
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor.reshape(N, H, W, F, S2)
        return tensor

    # Filter_adaptive_convolution
    def FAC(self, feat, filters, kernel_size):
        '''
            borrow from https://zhuanlan.zhihu.com/p/77718601
        '''
        N, C, H, W = feat.size()
        pad = (kernel_size - 1) // 2
        feat = self.unfold_and_permute(feat, kernel_size, 1, pad)

        filters = torch.cat((filters, filters, filters), dim = 1)
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, motion):
        for m in motion:
            feature = self.FAC(feature, m, self.kernel)
        return feature