import torch
from torch import nn
from torch.nn.parameter import Parameter
from sub_modules.component_block import *
import torch.nn.functional as F

#############################
#  The ECA layer is borrowed from https://github.com/BangguWu/ECANet
#############################
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        

#############################
# SFTGAN (pytorch version)
# Concept is borrowed from https://github.com/xinntao/SFTGAN/blob/master/pytorch_test/architectures.py
#############################

class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


#mix sft scale/shift
class motion_SFTLayer(nn.Module):
    def __init__(self, input_channel):
        super().__init__() 
        self.localize_motion = nn.Sequential(
                                nn.Conv2d(input_channel, input_channel, 1),
                                get_norm_layer('relu', input_channel))

    def adaIN(self, feature, motion):
        b, c, _, _ = feature.size()
        motion = motion.view(b, c, -1)

        # size of motion_tmean, motion_tvar (batch, channel * 2, 1, 1)
        motion_tmean = torch.mean(motion, dim = 2).unsqueeze(-1).unsqueeze(-1)
        motion_tvar = torch.var(motion, dim = 2).unsqueeze(-1).unsqueeze(-1)
        
        return feature * motion_tvar + motion_tmean

    def forward(self, feature, motion):
        motion = self.localize_motion(motion)
        feature = self.adaIN(feature, motion)
        return feature 


class motion_SFT(nn.Module):
    def __init__(self, input_channel, motion_channel):
        super().__init__() 
        self.localize_motion = nn.Sequential(
                                nn.Conv2d(motion_channel, input_channel, kernel_size = 3, padding=1),
                                get_norm_layer('relu', input_channel))

    def adaIN(self, feature, motion):
        b, c, _, _ = feature.size()
        motion = motion.view(b, c, -1)

        # size of motion_tmean, motion_tvar (batch, channel * 2, 1, 1)
        motion_tmean = torch.mean(motion, dim = 2).unsqueeze(-1).unsqueeze(-1)
        motion_tvar = torch.var(motion, dim = 2).unsqueeze(-1).unsqueeze(-1)
        
        return feature * motion_tvar #+ motion_tmean

    def forward(self, feature, motion):
        motion = self.localize_motion(motion)
        feature = self.adaIN(feature, motion)
        return feature 


class motion_var_SFT(nn.Module):
    def __init__(self, input_channel, motion_channel, ratio):
        super().__init__() 
        self.in8 = input_channel
        if ratio > 1:
            self.localize_motion = nn.Sequential(
                                nn.Upsample(scale_factor=ratio, mode='bilinear', align_corners=True),
                                get_norm_layer('leakyrelu', motion_channel),
                                nn.Conv2d(motion_channel, input_channel*9, kernel_size = 3, padding=1))
                                #get_norm_layer('leakyrelu', input_channel*9))
        else:
            self.localize_motion = nn.Sequential(
                                #nn.Conv2d(motion_channel, input_channel, kernel_size = 3, padding = 1),
                                get_norm_layer('leakyrelu', motion_channel),
                                #nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(motion_channel, input_channel*9, kernel_size = 3, padding = 1),
                                nn.LeakyReLU(0.2, inplace=True))#,
                                #get_norm_layer('leakyrelu', input_channel*9))

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
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, motion):
        motion = self.localize_motion(motion)
        x = self.FAC(feature, motion, 3)

        return x



class motion_Dynamic_filter(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt, input_channel, motion_channel, ratio):
        super().__init__() 
        self.kernel = 3
        self.ratio = ratio
        self.Leaky = nn.LeakyReLU(0.2)
        self.localize_motion = nn.Sequential(
                            StripPooling(opt, motion_channel, strip_size = 1),
                            nn.Conv2d(motion_channel, motion_channel, kernel_size=(self.kernel,self.kernel), stride=ratio, padding=1),
                            #Res_Block('leakyrelu', motion_channel),
                            Res_Block('leakyrelu', motion_channel),
                            nn.Conv2d(motion_channel, input_channel * (self.kernel **2), kernel_size=3, stride=1, padding=1),
                            Res_Block('leakyrelu', input_channel * (self.kernel **2)),
                            #Res_Block('leakyrelu', input_channel * (self.kernel **2)),
                            )   


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
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, motion):
        motion = self.localize_motion(motion)
        x = self.FAC(feature, motion, self.kernel)
        return self.Leaky(x)



class SP_residual_attention(nn.Module):
    def __init__(self, opt, input_channel, output_channel, ratio):
        super().__init__() 
        self.kernel = 5
        self.ratio = ratio
        self.motion_attention = nn.Sequential(
                                StripPooling(opt, input_channel, strip_size = 1),
                                nn.Conv2d(3, 3, kernel_size=(self.kernel,self.kernel), stride=ratio, padding=self.kernel//2),
                                Base_Res_Block(opt, 3),
                                nn.Conv2d(3, output_channel, kernel_size = 1, stride=1),
                                nn.Sigmoid()
                                )   

    def forward(self, feature, motion):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        motion_att = self.motion_attention(motion)

        return (motion_att * feature) + feature


class SP_reblur_attention_old(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 5
        self.per_pix_k_size = opt.per_pix_kernel
        self.reblur_attention = nn.Sequential(
                                StripPooling(opt, input_channel, strip_size = 1),
                                nn.Conv2d(input_channel, input_channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, input_channel, kernel_size = 1, stride=1),
                                #nn.Sigmoid(),        
                                )  
        '''
        if self.ratio > 1:
            self.reblur_filer = nn.Sequential(
                                nn.Upsample(scale_factor=self.ratio, mode='bilinear', align_corners=True),
                                StripPooling(opt, input_channel, strip_size = 1),
                                Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, 3 * self.per_pix_k_size ** 2, kernel_size = 1, stride=1),
                                #nn.LeakyReLU(0.2),
                                )
        else:
            self.reblur_filer = nn.Sequential(
                                StripPooling(opt, input_channel, strip_size = 1),
                                Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, 3 * self.per_pix_k_size ** 2, kernel_size = 1, stride=1),
                                #nn.LeakyReLU(0.2),
                                )
        '''

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        reblur_att = self.reblur_attention(feature)
        #reblur_filt = self.reblur_filer(reblur_att)
        decode = self.sigmoid(reblur_att) * feature

        return {'reblur_filter' : decode, 'decode_feature' : decode}


class SP_blur_prior(nn.Module):
    """
    Reference: https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py
    """
    def __init__(self, opt, in_channels, strip_size = 1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((strip_size, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, strip_size))
        self.kernel = 7

        self.conv_h = nn.Sequential(nn.Conv2d(3, in_channels, (1, 3), 1, (0, 1), bias=False))
        self.conv_v = nn.Sequential(nn.Conv2d(3, in_channels, (3, 1), 1, (1, 0), bias=False))
        self.conv_fusion = nn.Sequential(
                            Base_Res_Block(opt, in_channels),
                            Base_Res_Block(opt, in_channels),
                            nn.Conv2d(in_channels, in_channels, kernel_size = self.kernel, stride=1, padding = self.kernel//2),
                            )
        self.activation = nn.Tanh()


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
    def __init__(self, opt, in_channels, strip_size = 1):
        super().__init__()
        self.SP_scale = nn.Sequential(
                            SP_blur_prior(opt, in_channels),
                            #Base_Res_Block(opt, in_channels),
                            )

        self.SP_shift = nn.Sequential(
                            SP_blur_prior(opt, in_channels),
                            #Base_Res_Block(opt, in_channels),
                            )

        #self.instansnorm = nn.InstanceNorm2d(in_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x, blur):
        scale = self.SP_scale(blur)
        shift = self.SP_shift(blur)

        return  self.activation(x * scale + shift)


class SP_reblur_attention(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 5
        self.per_pix_k_size = opt.per_pix_kernel
        self.reblur_attention = nn.Sequential(
                                SP_module(opt, input_channel, strip_size = 1),
                                nn.Conv2d(input_channel, input_channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Base_Res_Block(opt, input_channel),
                                #nn.Conv2d(input_channel, input_channel, kernel_size = 1, stride=1),
                                #nn.Tanh(),
                                nn.Sigmoid(),        
                                )  
        #self.kernel = 1
        if self.ratio > 1:
            self.blur_prior_attention = SP_SFT(opt, self.per_pix_k_size ** 2, strip_size = 1)
            self.reblur_filer = nn.Sequential(
                                Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, (self.per_pix_k_size ** 2) * ratio * ratio, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                nn.PixelShuffle(ratio),
                                #SP_module(opt, (self.per_pix_k_size ** 2) * 1, strip_size = 1),
                                #nn.LeakyReLU(0.2),
                                #nn.Tanh()
                                #nn.Sigmoid(),
                                )

        else:
            self.blur_prior_attention = SP_SFT(opt, self.per_pix_k_size ** 2, strip_size = 1)
            self.reblur_filer = nn.Sequential(
                                #SP_SFT(opt, input_channel, strip_size = 1),
                                Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, (self.per_pix_k_size ** 2) * 1, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Base_Res_Block(opt, (self.per_pix_k_size ** 2) * 1),
                                #Base_Res_Block(opt, (self.per_pix_k_size ** 2) * 1),
                                #nn.Tanh(),
                                #SP_module(opt, (self.per_pix_k_size ** 2) * 1, strip_size = 1),
                                #nn.LeakyReLU(0.2),
                                )

        #print(self.reblur_filer)

    def forward(self, feature, blur):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        attention_feature = self.reblur_attention(feature) * feature

        reblur_filter = self.blur_prior_attention(self.reblur_filer(feature - attention_feature), blur)

        return {'reblur_filter' : reblur_filter, 'attention_feature' : attention_feature}



class SP_reblur_attention0(nn.Module):
    def __init__(self, opt, input_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = 5
        self.per_pix_k_size = opt.per_pix_kernel
        self.reblur_attention = nn.Sequential(
                                SP_module(opt, input_channel, strip_size = 1),
                                #nn.Conv2d(input_channel, input_channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, input_channel, kernel_size = 1, stride=1),
                                nn.Sigmoid(),        
                                )  

        if self.ratio > 1:
            #nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel*2, kernel_size=3, padding=1)
            self.reblur_filer = nn.Sequential(
                                nn.Upsample(scale_factor=self.ratio, mode='bilinear', align_corners=True),
                                nn.Conv2d(input_channel, (self.per_pix_k_size ** 2) * 1, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Base_Res_Block(opt, (self.per_pix_k_size ** 2) * 1),
                                #Base_Res_Block(opt, (self.per_pix_k_size ** 2) * 1),
                                SP_module_attention(opt, (self.per_pix_k_size ** 2) * 1, strip_size = 1),
                                #nn.LeakyReLU(0.2),
                                )

        else:
            self.reblur_filer = nn.Sequential(
                                #SP_module_attention(opt, input_channel, strip_size = 1),
                                #Base_Res_Block(opt, input_channel),
                                nn.Conv2d(input_channel, (self.per_pix_k_size ** 2) * 1, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Base_Res_Block(opt, (self.per_pix_k_size ** 2) * 1),
                                #Base_Res_Block(opt, (self.per_pix_k_size ** 2) * 1),
                                SP_module_attention(opt, (self.per_pix_k_size ** 2) * 1, strip_size = 1),
                                #nn.LeakyReLU(0.2),
                                )


    def forward(self, feature):
        '''
            input:
                motion size : batch, 27, H/4, W/4
        '''
        attention_feature = self.reblur_attention(feature) * feature

        reblur_filter = self.reblur_filer(feature - attention_feature)

        return {'reblur_filter' : reblur_filter, 'attention_feature' : attention_feature}


class motion_DF(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt):
        super().__init__() 
        self.kernel = opt.per_pix_kernel

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
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, motion):
        for m in motion:
            feature = self.FAC(feature, m, self.kernel)
        return feature


class blur_understanding_module(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt, in_channels = 3):
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
        print(feat.size(), filters.size())
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, motion):
        sharp = feature
        motion_list = []
        for m in motion:
            feature = self.FAC(feature, m, self.kernel)
        return feature


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



class blur_mask_understanding(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt, in_channels = 3):
        super().__init__() 
        self.kernel = opt.per_pix_kernel
        self.BlurMask = SpatialAttention()
        #self.mask = nn.Sequential(
        #            nn.Conv2d(self.kernel ** 2, 3, 3,  padding = 1),
        #            nn.Sigmoid()
        #            )
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

    def forward(self, feature, motion):
        motion_list = []
        for m in motion:
            mask = self.BlurMask(m)
            sharp = feature * (1-mask)
            blur = feature * mask
            blur = self.FAC(blur, m, self.kernel)
        self.mask  = {'blurmask' : mask, 'sharpmask':1-mask}
        return blur + sharp

class blur_mask_understandingv2(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt, in_channels = 3):
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

    def forward(self, feature, perpix, mask):
        origin = feature
        reblur = []
        for perpix_kernel, m in zip(perpix, mask):
            sharp = origin * (1-m)
            blur = origin * m
            reblur += [self.FAC(blur, perpix_kernel, self.kernel)]
        self.mask  = {'blurmask' : mask[0], 'sharpmask':1-mask[0]}
        return sum(reblur)/len(reblur)


class blur_mask_understandingv3(nn.Module):
    '''
        do localize motion for decoding 
    '''
    def __init__(self, opt, in_channels = 3):
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

    def forward(self, feature, perpix, mask):
        origin = feature
        sharp = origin * (1-mask[0])
        blur = origin * mask[0]        
        for perpix_kernel in perpix:
            blur = self.FAC(blur, perpix_kernel, self.kernel)
        self.mask  = {'blurmask' : mask[0], 'sharpmask':1-mask[0]}
        return blur + sharp


class blur_mask_understandingv4A(nn.Module):
    '''
        do localize motion for decoding, average
    '''
    def __init__(self, opt, in_channels = 3):
        super().__init__() 
        self.kernel = opt.per_pix_kernel
        self.channel = in_channels
        self.BlurMask = SpatialAttention()

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

    def forward(self, feature, perpix):
        reblur = []  
        for perpix_kernel in perpix:
            mask = self.BlurMask(perpix_kernel)
            sharp = feature * (1-mask)
            blur = feature * mask
            blur = self.FAC(blur, perpix_kernel, self.kernel)
            reblur += [sharp + blur]
        self.mask  = {'blurmask' : mask[0], 'sharpmask':1-mask[0]}
        return sum(reblur)/len(reblur)

class blur_mask_understandingv4(nn.Module):
    '''
        do localize motion for decoding, continue
    '''
    def __init__(self, opt, in_channels = 3):
        super().__init__() 
        self.kernel = opt.per_pix_kernel
        self.channel = in_channels
        self.BlurMask = SpatialAttention()

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

    def forward(self, feature, perpix):
        reblur = []  
        for perpix_kernel in perpix:
            mask = self.BlurMask(perpix_kernel)
            sharp = feature * (1-mask)
            blur = feature * mask
            blur = self.FAC(blur, perpix_kernel, self.kernel)
            #feature = sharp + blur
            reblur += [sharp + blur]
        self.mask  = {'blurmask' : mask[0], 'sharpmask':1-mask[0]}
        return sum(reblur)/len(reblur)

class blur_mask_understandingv5(nn.Module):
    '''
        do localize motion for decoding, continue
    '''
    def __init__(self, opt, in_channels = 3):
        super().__init__() 
        self.kernel = opt.per_pix_kernel
        self.channel = in_channels
        self.recurrent_times = opt.Recurrent_times
        self.BlurMask = nn.ModuleDict() 
        for i in range(self.recurrent_times):
            level = f'level_{i+1}'
            self.BlurMask[level] = SpatialAttention()

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

    def forward(self, feature, perpix):
        reblur = []  
        for i, perpix_kernel in enumerate(perpix):
            level = f'level_{i+1}'
            mask = self.BlurMask[level](perpix_kernel)
            sharp = feature * (1-mask)
            blur = feature * mask
            blur = self.FAC(blur, perpix_kernel, self.kernel)
            #feature = sharp + blur
            reblur += [sharp + blur]
        self.mask  = {'blurmask' : mask[0], 'sharpmask':1-mask[0]}
        return sum(reblur)/len(reblur)