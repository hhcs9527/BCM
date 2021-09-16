import torch
from torch import nn
from torch.nn.parameter import Parameter
from sub_modules.component_block import *
from DCNv2.dcn_v2 import DCN


class SP_motion_attention(nn.Module):
    def __init__(self, opt, input_channel, motion_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = opt.per_pix_kernel
        self.scale = 1

        #self.localize_motion = nn.Sequential(
        #                        StripPooling(opt, motion_channel, strip_size = 1),
        #                        #RES_FULL_ASPP_Block(opt, motion_channel, motion_channel, stride_len = 1),
        #                        Base_Res_Block(opt, motion_channel),
        #                        nn.Conv2d(motion_channel, input_channel * (self.kernel **2), kernel_size = 3, stride=ratio, padding=1),
        #                        nn.Sigmoid()
        #                        )   
        self.multi_SP = nn.ModuleDict()
        self.leaky = nn.LeakyReLU(0.2)
        kernel = 3
        
        for i in range(self.scale):
            level = f'level_{i}'
            self.multi_SP[level] = StripPooling(opt, motion_channel, strip_size = 2 * i + 1)

        self.motion_info = nn.Sequential(
                            DCN(motion_channel*self.scale, motion_channel*self.scale, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),
                            Residual_module(nn.Sequential(
                            #RES_FULL_ASPP_Block(opt, motion_channel*self.scale, motion_channel*self.scale, stride_len = 1),
                            DCN(motion_channel*self.scale, motion_channel*self.scale, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),
                            #Base_Res_Block(opt, motion_channel*self.scale)
                            )),
                            nn.Conv2d(motion_channel*self.scale, motion_channel, kernel_size = 3, stride=1, padding=1),
                            Base_Res_Block(opt, motion_channel),
                            nn.Conv2d(motion_channel, input_channel, kernel_size = 3, stride=ratio, padding=1))

        self.motion_scale = nn.Sequential(
                            #RES_FULL_ASPP_Block(opt, input_channel, input_channel, stride_len = 1),
                            Base_Res_Block(opt, input_channel),self.leaky)

        self.motion_shift = nn.Sequential(
                            #RES_FULL_ASPP_Block(opt, input_channel, input_channel, stride_len = 1),
                            Base_Res_Block(opt, input_channel),self.leaky)

        

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
        #attention = self.localize_motion(motion)
        #feature = self.FAC(feature, attention, self.kernel)

        motion_info = []
        for i in range(self.scale):
            level = f'level_{i}'
            motion_info.append(self.multi_SP[level](motion))

        localize_motion_info = self.motion_info(torch.cat((motion_info), dim = 1))
        scale = self.motion_scale(localize_motion_info)
        shift = self.motion_shift(localize_motion_info)
        feature = feature * scale + shift

        return  self.leaky(feature)


class SP_motion_afttention(nn.Module):
    def __init__(self, opt, input_channel, motion_channel, ratio):
        super().__init__() 
        self.ratio = ratio
        self.kernel = opt.per_pix_kernel
        self.scale = 3

        #self.localize_motion = nn.Sequential(
        #                        StripPooling(opt, motion_channel, strip_size = 1),
        #                        #RES_FULL_ASPP_Block(opt, motion_channel, motion_channel, stride_len = 1),
        #                        Base_Res_Block(opt, motion_channel),
        #                        nn.Conv2d(motion_channel, input_channel * (self.kernel **2), kernel_size = 3, stride=ratio, padding=1),
        #                        nn.Sigmoid()
        #                        )   
        self.multi_SP_scale = nn.ModuleDict()
        self.multi_SP_shift = nn.ModuleDict()
        self.leaky = nn.LeakyReLU(0.2)
        
        for i in range(self.scale):
            level = f'level_{i}'
            self.multi_SP_scale[level] = StripPooling(opt, motion_channel, strip_size = 2 * i + 1)
            self.multi_SP_shift[level] = StripPooling(opt, motion_channel, strip_size = 2 * i + 1)



        self.motion_scale = nn.Sequential(
                            nn.Conv2d(motion_channel*self.scale, motion_channel, kernel_size = 3, stride=1, padding=1),
                            #RES_FULL_ASPP_Block(opt, motion_channel, motion_channel, stride_len = 1),
                            Base_Res_Block(opt, motion_channel),
                            nn.Conv2d(motion_channel, input_channel, kernel_size = 3, stride=ratio, padding=1),
                            #RES_FULL_ASPP_Block(opt, input_channel, input_channel, stride_len = 1),
                            Base_Res_Block(opt, input_channel))

        self.motion_shift = nn.Sequential(
                            nn.Conv2d(motion_channel*self.scale, motion_channel, kernel_size = 3, stride=1, padding=1),
                            #RES_FULL_ASPP_Block(opt, motion_channel, motion_channel, stride_len = 1),
                            Base_Res_Block(opt, motion_channel),
                            nn.Conv2d(motion_channel, input_channel, kernel_size = 3, stride=ratio, padding=1),
                            #RES_FULL_ASPP_Block(opt, input_channel, input_channel, stride_len = 1),
                            Base_Res_Block(opt, input_channel))

        

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
        #attention = self.localize_motion(motion)
        #feature = self.FAC(feature, attention, self.kernel)

        shift = []
        scale = []
        for i in range(self.scale):
            level = f'level_{i}'
            scale.append(self.multi_SP_scale[level](motion))
            shift.append(self.multi_SP_shift[level](motion))

        scale = self.motion_scale(torch.cat((scale), dim = 1))
        shift = self.motion_shift(torch.cat((shift), dim = 1))
        feature = feature * scale + shift

        return  self.leaky(feature)


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
        x = self.FAC(feature, motion, self.kernel)
        return x