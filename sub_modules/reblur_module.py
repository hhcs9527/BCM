import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

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

class Reblurring_Module(nn.Module):
    '''
        Reblurring with blur information
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

    def forward(self, blur_info, perpix):
        reblur = []  
        for i, perpix_kernel in enumerate(perpix):
            level = f'level_{i+1}'
            mask = self.BlurMask[level](perpix_kernel)
            sharp = blur_info * (1-mask)
            blur = blur_info * mask
            blur = self.FAC(blur, perpix_kernel, self.kernel)
            reblur += [sharp + blur]

        self.mask  = {'blurmask' : mask[0], 'sharpmask':1-mask[0]}
        # averging for simulating the moving
        return sum(reblur)/len(reblur)