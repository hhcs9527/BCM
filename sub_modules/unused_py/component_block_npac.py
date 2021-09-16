import torch
import torch.nn as nn
import torch.nn.functional as F
from DCNv2.dcn_v2 import DCN
from sub_modules.attention_zoo import *

def get_norm_layer(norm_type, channel , padding=1, dilation=1):
    dilation_r = dilation
    if norm_type == 'batch':
        norm_layer = nn.Sequential(
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=padding),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=padding))
    elif norm_type == 'instance':
        norm_layer = nn.Sequential(
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=padding),
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=padding))
    elif norm_type == 'leaky':
        norm_layer = nn.Sequential(
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=1))
    else:
        norm_layer = nn.Sequential(
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channel, channel, dilation = dilation_r, kernel_size=3, padding=padding))
        
    return norm_layer


class StripPooling(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, opt, in_channels, strip_size = 1):
        super(StripPooling, self).__init__()
        kernel = 3
        # DCN as conv v/h
        #self.DCN = DCN(in_channels, in_channels, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),
        self.pool_h = nn.AdaptiveAvgPool2d((strip_size, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, strip_size))

        self.conv_h = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False))
        self.conv_v = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False))
        self.conv_h_deep = Base_Res_Block(opt, in_channels)
        self.conv_v_deep = Base_Res_Block(opt, in_channels) 
        self.conv_fusion = Base_Res_Block(opt, in_channels)


    def forward(self, x):
        _, _, h, w = x.size()
        x_horizon = F.interpolate(self.conv_h(self.pool_h(x)), (h, w), mode='bilinear', align_corners=True)
        x_vertical = F.interpolate(self.conv_v(self.pool_v(x)), (h, w), mode='bilinear', align_corners=True)
        x_horizon = self.conv_h_deep(x_horizon)
        x_vertical = self.conv_v_deep(x_vertical)
        return self.conv_fusion(x_horizon + x_vertical)



class Base_Res_Block(nn.Module):
    def __init__(self, opt, channels):
        super().__init__()
        self.res_block = get_norm_layer(opt.Norm, channels)

    def forward(self, x):
        return x + self.res_block(x)


class Residual_module(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.res_block = layer

    def forward(self, x):
        return x + self.res_block(x)


class Res_Block(nn.Module):
    def __init__(self, norm, channels):
        super().__init__()
        self.res_block = get_norm_layer(norm, channels)

    def forward(self, x):
        return x + self.res_block(x)



class RES_FULL_ASPP_Block(nn.Module):
    def __init__(self, opt, in_c, out_c, stride_len, kernel = 3):
        super().__init__()
        self.AR1 = nn.Conv2d(in_c, in_c, kernel_size=kernel, padding = 1, dilation = 1)
        self.AR2 = nn.Conv2d(in_c, in_c, kernel_size=kernel, padding = 2, dilation = 2)
        self.AR3 = nn.Conv2d(in_c, in_c, kernel_size=kernel, padding = 3, dilation = 3)
        self.AR4 = nn.Conv2d(in_c, in_c, kernel_size=kernel, padding = 4, dilation = 4)
        self.mix = nn.Sequential(
                                StripPooling(opt, in_c*4),
                                nn.Conv2d(in_c*4, out_c, kernel_size=kernel, padding = 1, dilation = 1),
                                nn.LeakyReLU(0.2, True)
                                )

    def forward(self, x):
        AR1 = self.AR1(x)
        AR2 = self.AR2(x)
        AR3 = self.AR3(x)
        AR4 = self.AR4(x)

        return self.mix(torch.cat((AR1, AR2, AR3, AR4), dim = 1)) + x


class RES_ASPP_Block(nn.Module):
    def __init__(self, in_c, out_c, stride_len):
        super().__init__()
        self.AR1 = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding = 1, dilation = 1)
        self.AR2 = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding = 2, dilation = 2)
        self.AR3 = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding = 3, dilation = 3)
        self.AR4 = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding = 4, dilation = 4)
        self.mix = nn.Conv2d(in_c*2, out_c, kernel_size=3, padding = 1, dilation = 1)

    def forward(self, x):
        AR1 = self.AR1(x)
        AR2 = self.AR2(x)
        AR3 = self.AR3(x)
        AR4 = self.AR4(x)

        return self.mix(torch.cat((AR1, AR2, AR3, AR4), dim = 1)) + x


class space_to_depth(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = nn.functional.unfold(x, self.ratio, stride = self.ratio)
        return unfolded_x.view(n, c * self.ratio ** 2, h // self.ratio, w // self.ratio)



class StripPooling_all(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, in_channels, pool_size, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),                      
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


