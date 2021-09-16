import torch
import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import random, time
import numpy as np
import cv2
import math
from itertools import chain 
from sub_modules.attention_zoo import *
from sub_modules.ConvLSTM_pytorch.convlstm import *
from sub_modules.generator import *
from sub_modules.origin_model import *

# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and not classname.find('Conv1d') != -1 and not classname.find('ConvLSTM') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size()).cuda()

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def define_G(opt, isGAN):

    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.G == 'Patch_Deblur_FPN_VAE':
        netG = Patch_Deblur_FPN_VAE(opt)

    elif opt.G == 'Patch_FPN_VAE':
        netG = Patch_FPN_VAE(opt)

    elif opt.G == 'Patch_FPN_SFT_fusion':
        netG = Patch_FPN_SFT_fusion(opt)

    elif opt.G == 'Patch_Motion_SFT_fusion':
        netG = Patch_Motion_SFT_fusion(opt)

    elif opt.G == 'Patch_Motion_per_pix':
        netG = Patch_Motion_per_pix(opt) 
    
    elif opt.G == 'Patch_FPN_DCN':
        netG = Patch_FPN_DCN(opt)

    elif opt.G == 'RDMPHN':
        netG = origin_RDMPHN_1_2_4(opt)

    elif opt.G == 'Patch_Motion_SP':
        netG = Patch_Motion_SP(opt)

    elif opt.G == 'Patch_Motion_BiFPN':
        netG = Patch_Motion_BiFPN(opt)

    elif opt.G == 'multi_scale_BiFPN':
        netG = multi_scale_BiFPN(opt)
    
    elif opt.G == 'double_reblur_DMPHN_1_2_4':
        netG = double_reblur_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRM_DMPHN_1_2_4_8':
        netG = BRRM_DMPHN_1_2_4_8(opt)
    
    elif opt.G == 'DMPHN':
        netG = origin_DMPHN_1_2_4(opt)

    elif opt.G == 'BRR_DMPHN_1_2_4':
        netG = BRR_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRM_DMPHN_1_2_4':
        netG = BRRM_DMPHN_1_2_4(opt)

    elif opt.G == 'BRM_DMPHN_1_2_4':
        netG = BRM_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRML_DMPHN_1_2_4':
        netG = BRRML_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMv2_DMPHN_1_2_4':
        netG = BRRMv2_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMv3_DMPHN_1_2_4':
        netG = BRRMv3_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRDCN_DMPHN_1_2_4':
        netG = BRRDCN_DMPHN_1_2_4(opt)

    elif opt.G == 'BER_DMPHN_1_2_4':
        netG = BER_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMALL_DMPHN_1_2_4':
        netG = BRRMALL_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMALL_DMPHN_1_2_4_8':
        netG = BRRMALL_DMPHN_1_2_4_8(opt)

    elif opt.G == 'BRRMv4_DMPHN_1_2_4':
        netG = BRRMv4_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMv5_DMPHN_1_2_4':
        netG = BRRMv5_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMv4_DMPHN_1_2_4_8':
        netG = BRRMv4_DMPHN_1_2_4_8(opt)

    elif opt.G == 'BRRMv5_DMPHN_1_2_4_8':
        netG = BRRMv5_DMPHN_1_2_4_8(opt)

    elif opt.G == 'BRRMv5ALL_DMPHN_1_2_4':
        netG = BRRMv5ALL_DMPHN_1_2_4(opt)

    elif opt.G == 'BRRMv4ALL_DMPHN_1_2_4_8':
        netG = BRRMv4ALL_DMPHN_1_2_4_8(opt)

    #else:
    #    netG = RFDB_reblur_DMPHN_1_2_4(opt)

    return netG.apply(weights_init).to(use_gpu)


def define_content_D(opt):
    '''
        Follow the design of Encoder in DMPHN
    '''
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.content_D == 'DMPHN':
        netD = Encoder(opt)
    return netD.to(use_gpu).apply(weights_init)


def define_image_D(opt):
    ''' 
        Follow the design of multi-path part of DMPHN, to get the detail part of the whole image 
    '''
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.image_D == 'multi_patch':
        netD = Encoder(opt)
    return netD.to(use_gpu).apply(weights_init)


def define_alignment_module(opt):
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
    alignment_module = PCD_alignment(opt)

    return alignment_module.to(use_gpu).apply(weights_init)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class origin_RDMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = RDMPHN_1_2_4(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur = self.DMPHN(input)['deblur']

        return {'deblur': [deblur], 'reblur' : {'sharp_reblur': input}}

class reblur_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN =  DMPHN_1_2_4_reblur(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        reblur = None
        if self.istrain:
            reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : reblur }


class double_reblur_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_double_reblur(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        reblur = None
        if self.istrain:
            reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : reblur }


class origin_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur = self.DMPHN(input)
        #reblur = None
        #if self.istrain:
        #    reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])

        return {'deblur': [deblur], 'reblur' : input}


class RFDB_reblur_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_RFDB_reblur_1_2_4(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        reblur = None
        if self.istrain:
            #print(len(deblur_dict['reblur_filter']))
            reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : reblur}

class BRR_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRR(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRR_DMPHN_1_2_4_8(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_8_BRR(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BER_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BER(opt)
        self.reblur_module = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRM_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BR(opt)
        self.reblur_module = blur_mask_understanding(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRML_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRRML(opt)
        self.reblur_module = blur_mask_understanding(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRRM_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRRMv1(opt)
        self.reblur_module = blur_mask_understanding(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRM_DMPHN_1_2_4_8(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_8_BRR(opt)
        self.reblur_module = blur_mask_understanding(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRMv2_DMPHN_1_2_4(nn.Module):
    '''
        1. add SP reblur to get mask
        2. per-pixel kernel = 9
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRRM(opt)
        self.reblur_module = blur_mask_understandingv2(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'], deblur_dict['blurmask'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'], deblur_dict['blurmask'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRRMv3_DMPHN_1_2_4(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 7
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRRM(opt)
        self.reblur_module = blur_mask_understandingv3(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
        '''
            use self.recurrent_time - 1 times of self.ConvLSTM
            return :
                self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
        '''
        self.reblur_filter = motion_feature
        hidden_state = None
        for i in range(self.recurrent_times - 1):
            level = f'level_{i+1}'
            motion = self.reblur_filter[i]
            layer_outputs, hidden_state = self.convlstm[level](torch.unsqueeze(self.reblur_filter[i], 1), hidden_state)
            self.reblur_filter.append(torch.squeeze(layer_outputs[0], 1))
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter, deblur_dict['blurmask'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter, deblur_dict['blurmask'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRRMv4_DMPHN_1_2_4(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRR(opt)
        self.reblur_module = blur_mask_understandingv4(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
        '''
            use self.recurrent_time - 1 times of self.ConvLSTM
            return :
                self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
        '''
        self.reblur_filter = motion_feature
        hidden_state = None
        for i in range(self.recurrent_times - 1):
            level = f'level_{i+1}'
            motion = self.reblur_filter[i]
            layer_outputs, hidden_state = self.convlstm[level](torch.unsqueeze(self.reblur_filter[i], 1), hidden_state)
            self.reblur_filter.append(torch.squeeze(layer_outputs[0], 1))
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRMv5_DMPHN_1_2_4(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRR(opt)
        self.reblur_module = blur_mask_understandingv5(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
        '''
            use self.recurrent_time - 1 times of self.ConvLSTM
            return :
                self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
        '''
        self.reblur_filter = motion_feature
        hidden_state = None
        for i in range(self.recurrent_times - 1):
            level = f'level_{i+1}'
            motion = self.reblur_filter[i]
            layer_outputs, hidden_state = self.convlstm[level](torch.unsqueeze(self.reblur_filter[i], 1), hidden_state)
            self.reblur_filter.append(torch.squeeze(layer_outputs[0], 1))
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRMv5_DMPHN_1_2_4_8(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_8_BRR(opt)
        self.reblur_module = blur_mask_understandingv5(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
        '''
            use self.recurrent_time - 1 times of self.ConvLSTM
            return :
                self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
        '''
        self.reblur_filter = motion_feature
        hidden_state = None
        for i in range(self.recurrent_times - 1):
            level = f'level_{i+1}'
            motion = self.reblur_filter[i]
            layer_outputs, hidden_state = self.convlstm[level](torch.unsqueeze(self.reblur_filter[i], 1), hidden_state)
            self.reblur_filter.append(torch.squeeze(layer_outputs[0], 1))
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRMv4_DMPHN_1_2_4_8(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_8_BRR(opt)
        self.reblur_module = blur_mask_understandingv4(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
        '''
            use self.recurrent_time - 1 times of self.ConvLSTM
            return :
                self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
        '''
        self.reblur_filter = motion_feature
        hidden_state = None
        for i in range(self.recurrent_times - 1):
            level = f'level_{i+1}'
            motion = self.reblur_filter[i]
            layer_outputs, hidden_state = self.convlstm[level](torch.unsqueeze(self.reblur_filter[i], 1), hidden_state)
            self.reblur_filter.append(torch.squeeze(layer_outputs[0], 1))
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRRMv5ALL_DMPHN_1_2_4(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRRALL(opt)
        self.reblur_module = blur_mask_understandingv5(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
            '''
                use self.recurrent_time - 1 times of self.ConvLSTM
                return :
                    self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
            '''
            hidden_state = None
            self.reblur_filter = []
            for j in range(len(motion_feature)):
                motionInfo = motion_feature[j]
                self.reblur_filter.append(motionInfo)
                for i in range(self.recurrent_times - 1):
                    level = f'level_{i+1}'
                    motion = motionInfo
                    motionInfo, hidden_state = self.convlstm[level](torch.unsqueeze(motionInfo, 1), hidden_state)
                    self.reblur_filter.append(torch.squeeze(motionInfo[0], 1))
                    motionInfo = torch.squeeze(motionInfo[0], 1)
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

class BRRMv4ALL_DMPHN_1_2_4_8(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_8_BRRALL(opt)
        self.reblur_module = blur_mask_understandingv4(opt)
        self.opt = opt
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
            '''
                use self.recurrent_time - 1 times of self.ConvLSTM
                return :
                    self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
            '''
            hidden_state = None
            self.reblur_filter = []
            for j in range(len(motion_feature)):
                motionInfo = motion_feature[j]
                self.reblur_filter.append(motionInfo)
                for i in range(self.recurrent_times - 1):
                    level = f'level_{i+1}'
                    motion = motionInfo
                    motionInfo, hidden_state = self.convlstm[level](torch.unsqueeze(motionInfo, 1), hidden_state)
                    self.reblur_filter.append(torch.squeeze(motionInfo[0], 1))
                    motionInfo = torch.squeeze(motionInfo[0], 1)
    
    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRRMALL_DMPHN_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_BRRALL2080(opt)
        self.reblur_module = blur_mask_understanding(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}


class BRRMALL_DMPHN_1_2_4_8(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DMPHN = DMPHN_1_2_4_8_BRRALL(opt)
        self.reblur_module = blur_mask_understanding(opt)
        self.opt = opt

    def forward(self, input, sharp):
        '''
            return 
                1.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
        '''
        self.istrain = self.opt.isTrain
        deblur_dict = self.DMPHN(input)
        sharp_reblur, deblur_reblur = None, None
        #if self.istrain:
        sharp_reblur = self.reblur_module(sharp, deblur_dict['reblur_filter'])
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], deblur_dict['reblur_filter'])

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

if __name__ == '__main__':
    print('hello')