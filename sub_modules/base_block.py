import torch.nn as nn
import torch
import torch.nn.functional as F
from sub_modules.ConvLSTM_pytorch.convlstm import *

# get short range relation
class per_pix_convolution(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt, in_channels):
        super().__init__() 
        self.kernel = opt.per_pix_kernel
        self.channel = in_channels

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
        filters = filters.repeat(1, self.channel, 1, 1)
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, attention):
        origin = feature
        feature = self.FAC(feature, attention, self.kernel)
        return origin + feature 


class StripPooling(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, in_channels, pool_size = [20, 12]):
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

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=True)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# get long range relation
class Blur_Capture_module(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, opt, in_channels, strip_size = 1):
        super().__init__()
        self.kernel = 1
        self.pool_h = nn.AdaptiveAvgPool2d((strip_size, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, strip_size))
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size = self.kernel, stride=1, padding = self.kernel//2),
                    ) 
        self.per_pix_kernel = opt.per_pix_kernel
        self.conv_h = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False))
        self.conv_v = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False))
        self.conv_fusion = nn.Sequential(
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride=1, padding = 3//2),
                            nn.LeakyReLU(0.2),
                            #nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride=1, padding = 3//2),
                            )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv(x)
        x_horizon = F.interpolate(self.conv_h(self.pool_h(x)), (h, w), mode='bilinear', align_corners=True)
        x_vertical = F.interpolate(self.conv_v(self.pool_v(x)), (h, w), mode='bilinear', align_corners=True)

        return self.conv_fusion(x + x_horizon + x_vertical)

# as a block
class BCM_block(nn.Module):
    def __init__(self, opt, channel):
        super().__init__()
        self.per_pix_channel = opt.per_pix_kernel **2
        self.channel = channel
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.2)
        self.act = nn.LeakyReLU(0.2)

        # we have 4 level conv(including last one)
        self.conv3x3_lv1 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        self.relu,
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        )

        self.conv3x3_lv2 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        self.relu,
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        )
        self.conv_blur_lv2 = nn.Sequential(
                        Blur_Capture_module(opt, self.channel),
                        )
        self.conv1x1_lv2 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 3, padding=1),
                        self.act,
                        )

    def forward(self, input):
        '''
            concept here is to do the following operation
            
                                                   --> reblur
                                                   |                   
                                           -> conv blur -> x - blur -
                                           |                         |
                                           |                         V
            x -> conv -> relu -> conv -------> conv -> relu -> conv --->

        '''
        self.reblur_filter = []
        sharpen1 = input
        conv1 = self.conv3x3_lv1(sharpen1) + sharpen1

        blur2 = self.conv_blur_lv2(conv1)
        self.reblur_filter.append(self.conv1x1_lv2(blur2))
        sharpen2 = conv1 - blur2
        result = self.conv3x3_lv2(sharpen2) + sharpen2 

        return {'encode_feature' : result, 'reblur_filter' : self.reblur_filter, 'blur':blur2, 'conv':conv1, 'sharpen':sharpen2}

if __name__ == "__main__":
    x = 0