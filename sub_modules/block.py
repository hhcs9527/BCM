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

#####
# borrow from:
# https://github.com/njulj/RFDN/blob/master/block.py
#####

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return x * m

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = x + self.sub(x)
        return self.act(output)

# as a block
class RFD_block_cat(nn.Module):
    def __init__(self, channel, compression_ratio = 2):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.channel = channel
        self.compress_channel = self.channel // self.compression_ratio

        # we have 4 level conv(including last one)
        self.conv3x3_lv1 = nn.Conv2d(self.channel, self.compress_channel, kernel_size = 3, padding=1)
        self.conv1x1_lv1 = nn.Conv2d(self.channel, self.compress_channel, kernel_size = 1)

        self.conv3x3_lv2 = ShortcutBlock(nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        ))
        self.conv1x1_lv2 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 1),
                        nn.LeakyReLU(0.2),
                        )

        self.conv3x3_lv3 = ShortcutBlock(nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        ))
        self.conv1x1_lv3 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 1),
                        nn.LeakyReLU(0.2),
                        )

        self.conv3x3_lv4 = ShortcutBlock(nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        ))
        k = 3
        self.information_shuffle_layer = nn.Sequential(
                        nn.Conv2d(self.compress_channel * 4, self.channel, kernel_size = 1),
                        ShortcutBlock(nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = k, padding=k//2),)),
                        )

    def forward(self, input):
        x = input 
        distilled_c1 = self.conv1x1_lv1(x)
        refine_lv1 = self.conv3x3_lv1(x)

        distilled_c2 = self.conv1x1_lv2(refine_lv1)
        refine_lv2 = self.conv3x3_lv2(refine_lv1)

        distilled_c3 = self.conv1x1_lv3(refine_lv2)
        refine_lv3 = self.conv3x3_lv3(refine_lv2)

        distilled_c4 = self.conv3x3_lv4(refine_lv3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
        out_fused = self.information_shuffle_layer(out)
        return out_fused + input

# as a block
class RFD_block(nn.Module):
    def __init__(self, channel, compression_ratio = 2):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.channel = channel
        self.compress_channel = self.channel // self.compression_ratio

        # we have 4 level conv(including last one)
        self.conv3x3_lv1 = nn.Conv2d(self.channel, self.compress_channel, kernel_size = 3, padding=1)
        self.conv1x1_lv1 = nn.Conv2d(self.channel, self.channel, kernel_size = 1)

        self.conv3x3_lv2 = ShortcutBlock(nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        ))
        self.conv1x1_lv2 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.channel, kernel_size = 1),
                        nn.LeakyReLU(0.2),
                        )

        self.conv3x3_lv3 = ShortcutBlock(nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        ))
        self.conv1x1_lv3 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.channel, kernel_size = 1),
                        nn.LeakyReLU(0.2),
                        )

        self.conv3x3_lv4 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.channel, kernel_size = 3, padding=1),
                        )
        k = 1
        self.information_shuffle_layer = ShortcutBlock(nn.Sequential(
                        ShortcutBlock(
                        nn.Conv2d(self.channel, self.channel, kernel_size = k, padding=k//2)),
                        #nn.LeakyReLU(0.2),
                        ShortcutBlock(
                        nn.Conv2d(self.channel, self.channel, kernel_size = k, padding=k//2)),
                        ))

    def forward(self, input):
        distilled_c1 = self.conv1x1_lv1(input)
        refine_lv1 = self.conv3x3_lv1(input)

        distilled_c2 = self.conv1x1_lv2(refine_lv1)
        refine_lv2 = self.conv3x3_lv2(refine_lv1)

        distilled_c3 = self.conv1x1_lv3(refine_lv2)
        refine_lv3 = self.conv3x3_lv3(refine_lv2)

        distilled_c4 = self.conv3x3_lv4(refine_lv3)

        out = distilled_c1 + distilled_c2 + distilled_c3 + distilled_c4
        out_fused = self.information_shuffle_layer(out)
        return out_fused + input

# as a block
class RFD_block_reblur_then(nn.Module):
    def __init__(self, opt, channel, compression_ratio = 1):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.per_pix_channel = opt.per_pix_kernel **2
        self.channel = channel
        self.compress_channel = self.channel // self.compression_ratio
        self.relu = nn.ReLU()
        self.act = nn.Tanh()

        # we have 4 level conv(including last one)
        self.conv3x3_lv1 = nn.Sequential(
                        nn.Conv2d(self.channel, self.compress_channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv1x1_lv1 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )

        self.conv3x3_lv2 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv1x1_lv2 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )

        self.conv3x3_lv3 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.compress_channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv1x1_lv3 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )

        self.conv3x3_lv4 = nn.Sequential(
                        nn.Conv2d(self.compress_channel, self.channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )

    def forward(self, input):
        self.reblur_filter = []
        conv1 = self.conv3x3_lv1(input)
        self.reblur_filter.append(self.conv1x1_lv1(input - conv1))

        conv2 = self.conv3x3_lv2(conv1)
        self.reblur_filter.append(self.conv1x1_lv2(conv1 - conv2))

        conv3 = self.conv3x3_lv3(conv2)
        self.reblur_filter.append(self.conv1x1_lv3(conv2 - conv3))

        result = self.conv3x3_lv4(conv3)
        return {'encode_feature' : result, 'reblur_filter' : self.reblur_filter}

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

# as a block
class RFD_block_reblur_SRB(nn.Module):
    def __init__(self, opt, channel, compression_ratio = 1):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.per_pix_channel = opt.per_pix_kernel **2
        self.channel = channel
        self.compress_channel = self.channel // self.compression_ratio
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.2)
        self.act = nn.Tanh()

        # we have 4 level conv(including last one)
        self.conv3x3_lv1 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv_blur_lv1 = nn.Sequential(
                        StripPooling(self.channel),
                        #nn.Conv2d(self.channel, self.channel, kernel_size = 1),
                        self.leaky,
                        )
        self.conv1x1_lv1 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )
        

        self.conv3x3_lv2 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv_blur_lv2 = nn.Sequential(
                        StripPooling(self.channel),
                        #nn.Conv2d(self.channel, self.channel, kernel_size = 1),
                        self.leaky,
                        )
        self.conv1x1_lv2 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )

        self.conv3x3_lv3 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv_blur_lv3 = nn.Sequential(
                        StripPooling(self.channel),
                        #nn.Conv2d(self.channel, self.channel, kernel_size = 1),
                        self.leaky,
                        )
        self.conv1x1_lv3 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )

        self.conv3x3_lv4 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        #SP_conv(opt, channel),
                        self.relu,
                        )
        self.conv_blur_lv4 = nn.Sequential(
                        StripPooling(self.channel),
                        #nn.Conv2d(self.channel, self.channel, kernel_size = 1),
                        self.leaky,
                        )
        self.conv1x1_lv4 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 1),
                        self.act,
                        )

    def forward(self, input):
        '''
            concept here is to do the following operation
            
            1. x -> blur of x
            2. conv(x - blur of x)
            3. so on so forth
        '''
        self.reblur_filter = []

        blur1 = self.conv_blur_lv1(input)
        self.reblur_filter.append(self.conv1x1_lv1(blur1))
        conv1 = self.conv3x3_lv1(input) + input - blur1

        blur2 = self.conv_blur_lv2(conv1)
        self.reblur_filter.append(self.conv1x1_lv2(blur2))
        conv2 = self.conv3x3_lv2(conv1) + conv1 - blur2

        blur3 = self.conv_blur_lv3(conv2)
        self.reblur_filter.append(self.conv1x1_lv3(blur3))
        conv3 = self.conv3x3_lv3(conv2) + conv2 - blur3

        blur4 = self.conv_blur_lv4(conv3)
        self.reblur_filter.append(self.conv1x1_lv4(blur4))
        result = self.conv3x3_lv4(conv3) + conv3 - blur4

        return {'encode_feature' : result, 'reblur_filter' : self.reblur_filter}

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
class SP_blur_conv(nn.Module):
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
class RFD_block_reblur(nn.Module):
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
                        SP_blur_conv(opt, self.channel),
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


# as a block
class RFD_block_reblurDCN(nn.Module):
    def __init__(self, opt, channel):
        super().__init__()
        self.per_pix_channel = (opt.per_pix_kernel **2)*2
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
                        SP_blur_conv(opt, self.channel),
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



# as a block
class RFD_block_reblur_multi_scale(nn.Module):
    def __init__(self, opt, channel, ratio = 1):
        super().__init__()
        self.ratio = ratio
        self.per_pix_channel = opt.per_pix_kernel ** 2
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
                        SP_blur_conv(opt, self.channel),
                        )
        self.k = 3
        if self.ratio == 1:
            self.conv1x1_lv2 = nn.Sequential(
                            nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 3, padding=1),
                            self.act,
                            )
        else:
            self.conv1x1_lv2 = nn.Sequential(
                            nn.Upsample(scale_factor = self.ratio, mode='bilinear', align_corners=True),
                            nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 3, padding=1),
                            self.act,
                            )

    def forward(self, input):
        '''
            concept here is to do the following operation
            
                                                   --> reblur
                                                   |                   
                                           -> conv blur -> x * (1-blur) 
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


# get long range relation
class SP_blur_mask(nn.Module):
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
                            nn.Conv2d(in_channels, 1, kernel_size = 3, stride=1, padding = 3//2),
                            nn.Sigmoid(),
                            )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv(x)
        x_horizon = F.interpolate(self.conv_h(self.pool_h(x)), (h, w), mode='bilinear', align_corners=True)
        x_vertical = F.interpolate(self.conv_v(self.pool_v(x)), (h, w), mode='bilinear', align_corners=True)

        return self.conv_fusion(x + x_horizon + x_vertical)

# as a block
class RFD_block_reblur_mask(nn.Module):
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
                        SP_blur_conv(opt, self.channel),
                        )
        self.conv_perpix_blur = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 3, padding=1),
                        self.act,
                        )
        self.conv_blur_mask = nn.Sequential(
                        SP_blur_mask(opt, self.channel),
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
        self.reblur_filter.append(self.conv_perpix_blur(blur2))
        blurmask = self.conv_blur_mask(blur2)
        sharpen2 = conv1 - blur2
        result = self.conv3x3_lv2(sharpen2) + sharpen2 

        return {'encode_feature' : result, 'reblur_filter' : self.reblur_filter, 'blur':blur2, 'conv':conv1, 'sharpen':sharpen2, 'blurmask' : blurmask}


# get long range relation
class SP_blur_conv_old(nn.Module):
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
class RFD_block_reblur_old(nn.Module):
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
        self.conv_blur_lv1 = nn.Sequential(
                        SP_blur_conv_old(opt, self.channel),
                        )
        self.conv1x1_lv1 = nn.Sequential(
                        nn.Conv2d(self.channel, self.per_pix_channel, kernel_size = 3, padding=1),
                        self.act,
                        )


        self.conv3x3_lv2 = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        self.relu,
                        nn.Conv2d(self.channel, self.channel, kernel_size = 3, padding=1),
                        )
        self.conv_blur_lv2 = nn.Sequential(
                        SP_blur_conv_old(opt, self.channel),
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

        return {'encode_feature' : result, 'reblur_filter' : self.reblur_filter, 'blur':blur2, 'conv':blur2, 'sharpen':sharpen2}



# get long range relation
class SP(nn.Module):
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
class RFD_block(nn.Module):
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
                        SP_blur_conv(opt, self.channel),
                        )
        self.conv_perpix_blur = nn.Sequential(
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
        self.reblur_filter.append(self.conv_perpix_blur(blur2))
        sharpen2 = conv1 - blur2
        result = self.conv3x3_lv2(sharpen2) + sharpen2 

        return {'encode_feature' : result, 'reblur_filter' : self.reblur_filter, 'blur':blur2, 'conv':conv1, 'sharpen':sharpen2}



if __name__ == "__main__":
    x = 0