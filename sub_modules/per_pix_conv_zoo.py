import torch
from torch import nn
import torch.nn.functional as F

# for reblurring the sharp image
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
        weight = self.weight_permute_reshape(filters, C, kernel_size**2)
        
        output = feat * weight
        output = output.sum(-1)
        output = output.permute(0,3,1,2)
        return output

    def forward(self, feature, motion):
        for m in motion:
            feature = self.FAC(feature, m, self.kernel)
        return feature
        
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