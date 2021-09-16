import torch
from torch import nn
from torch.nn.parameter import Parameter
from sub_modules.component_block import *
from sub_modules.block import *
import torch.nn.functional as F
from sub_modules.per_pix_conv_zoo import *

# get long range relation
class SP_conv(nn.Module):
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
                            nn.Conv2d(in_channels, self.per_pix_kernel * self.per_pix_kernel, kernel_size = self.kernel, stride=1, padding = self.kernel//2),
                            nn.Tanh(),
                            )
        self.per_pix_conv = per_pix_convolution(opt, in_channels)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv(x)
        x_horizon = F.interpolate(self.conv_h(self.pool_h(x)), (h, w), mode='bilinear', align_corners=True)
        x_vertical = F.interpolate(self.conv_v(self.pool_v(x)), (h, w), mode='bilinear', align_corners=True)

        return self.per_pix_conv(x, self.conv_fusion(x_horizon + x_vertical))

# get short range relation
class local_conv(nn.Module):
    def __init__(self, opt, in_channels, conv_size = 3):
        super().__init__()
        self.kernel = conv_size
        self.per_pix_kernel = opt.per_pix_kernel
        self.conv = nn.Sequential(
                            nn.Conv2d(in_channels, self.per_pix_kernel * self.per_pix_kernel, kernel_size = self.kernel, stride=1, padding = self.kernel//2),
                            #nn.Conv2d(self.per_pix_kernel * self.per_pix_kernel, self.per_pix_kernel * self.per_pix_kernel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                            nn.Tanh(),
                            )
        self.per_pix_conv = per_pix_convolution(opt, in_channels)

    def forward(self, x):
        local_attention = self.conv(x)
        return self.per_pix_conv(x, local_attention)

# for maintaining sharp feature, generate general_attention map
class sp_per_pix_attention(nn.Module):
    def __init__(self, opt, in_channels, strip_size = 1):
        super().__init__()
        self.kernel = 1
        self.long_attention = SP_conv(opt, in_channels)
        self.short_attention = local_conv(opt, in_channels)
        self.conv_out = nn.Sequential( 
                            nn.Conv2d(in_channels, in_channels, kernel_size = self.kernel, stride=1, padding = self.kernel//2),
                            nn.LeakyReLU(0.2)
                            )                    
    def forward(self, x):
        short_range_information = self.short_attention(x)
        long_range_information = self.long_attention(x)
        return x * long_range_information + short_range_information



# find sharp feature, blur feature simultaneously
class BS_attention(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 1
        self.per_pix_k_size = opt.per_pix_kernel
        self.blur_resize = nn.Conv2d(input_channel//2 ,input_channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2)
        self.sharp_resize = nn.Conv2d(input_channel//2 ,input_channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2)
        self.sharp_attention = nn.Sequential(
                                sp_per_pix_attention(opt, input_channel),       
                                )  
        self.kernel = 1
        self.num_filters = 1
        if self.ratio > 1:
            self.reblur_filer = nn.Sequential(
                                nn.Upsample(scale_factor = self.ratio, mode='bilinear', align_corners=True),
                                nn.Conv2d(input_channel//2 ,(self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #nn.Sigmoid(),
                                nn.Tanh(),
                                #nn.LeakyReLU(0.2),
                                )
        else:
            self.reblur_filer = nn.Sequential(
                                nn.Conv2d(input_channel//2, (self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #nn.Sigmoid(),
                                nn.Tanh(),
                                #nn.LeakyReLU(0.2),
                                )

    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        _, c, _, _ = feature.size()
        sharp, blur = torch.split(feature, c//2, dim = 1)

        attention_feature = self.sharp_attention(self.sharp_resize(sharp))
        reblur_filter = self.reblur_filer(blur)

        return {'reblur_filter' : reblur_filter, 'attention_feature' : feature - attention_feature}


# find sharp feature, blur feature simultaneously
class reblur_global_attention(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 1
        self.per_pix_k_size = opt.per_pix_kernel
        self.sharp_attention = nn.Sequential(
                                SP_conv(opt, input_channel),
                                )  
        self.kernel = 1
        self.num_filters = 1
        if self.ratio > 1:
            self.reblur_filer = nn.Sequential(
                                nn.Upsample(scale_factor = self.ratio, mode='bilinear', align_corners=True),
                                nn.Conv2d(input_channel ,(self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                nn.Tanh(),
                                )
        else:
            self.reblur_filer = nn.Sequential(
                                nn.Conv2d(input_channel, (self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                nn.Tanh(),
                                )

    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        _, c, _, _ = feature.size()
        sharp, blur = torch.split(feature, c//2, dim = 1)

        attention_feature = self.sharp_attention(sharp)
        reblur_filter = self.reblur_filer(blur)

        return {'reblur_filter' : reblur_filter, 'attention_feature' : attention_feature + feature}


# find sharp feature, blur feature simultaneously
class reblur_local_attention(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 1
        self.per_pix_k_size = opt.per_pix_kernel
        self.sharp_attention = nn.Sequential(
                                local_conv(opt, input_channel, conv_size = 5),       
                                )  
        self.kernel = 3
        self.num_filters = 1
        if self.ratio > 1:
            self.reblur_filer = nn.Sequential(
                                nn.Upsample(scale_factor = self.ratio, mode='bilinear', align_corners=True),
                                nn.Conv2d(input_channel ,(self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                nn.Conv2d((self.per_pix_k_size ** 2) * self.num_filters ,(self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #nn.ReLU(),
                                nn.Tanh(),
                                )
        else:
            self.reblur_filer = nn.Sequential(
                                nn.Conv2d(input_channel, (self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                nn.Conv2d((self.per_pix_k_size ** 2) * self.num_filters ,(self.per_pix_k_size ** 2) * self.num_filters, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #nn.ReLU(),
                                nn.Tanh(),
                                )

    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        attention_feature = self.sharp_attention(feature)

        reblur_filter = self.reblur_filer(feature - attention_feature)

        return {'reblur_filter' : reblur_filter, 'attention_feature' : attention_feature}


# find sharp feature, blur feature simultaneously
class global_attention(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 1
        self.per_pix_k_size = opt.per_pix_kernel
        self.sharp_attention = nn.Sequential(
                                SP_conv(opt, input_channel),
                                )  
    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        attention_feature = self.sharp_attention(feature)
        reblur_filter = []

        return {'reblur_filter' : reblur_filter, 'attention_feature' : attention_feature}
