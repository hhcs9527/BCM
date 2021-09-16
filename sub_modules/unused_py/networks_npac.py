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
from sub_modules.encoder_zoo import *
from sub_modules.decoder_zoo import *
from sub_modules.guide_feature_extractor import *
#from sub_modules.alignment_zoo import *
from sub_modules.attention_zoo import *
from sub_modules.ConvLSTM_pytorch.convlstm import *
#from sub_modules.generator import *
#from sub_modules.origin_model import *

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
    use_gpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
    
    if opt.G == 'Patch_Motion_per_pix':
        netG = Patch_Motion_SP(opt)
    return netG.apply(weights_init).to(use_gpu)


def define_content_D(opt):
    '''
        Follow the design of Encoder in DMPHN
    '''
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.content_D == 'DMPHN':
        netD = Localize_DMPHN_1_2_4_8(opt)
    return netD.to(use_gpu).apply(weights_init)


def define_image_D(opt):
    ''' 
        Follow the design of multi-path part of DMPHN, to get the detail part of the whole image 
    '''
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.image_D == 'multi_patch':
        netD = Localize_DMPHN_1_2_4_8(opt)
    return netD.to(use_gpu).apply(weights_init)



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class Patch_Motion_SP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_SP_motion(opt)
        self.encoder = SP_FPN_encoder(opt) 
        self.decoder = SP_FPN_decoder(opt)
        self.reblur_net = motion_DF(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        FPN_info, pred_bifpn = self.encoder(input, motion_dict['motion_prior'])
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        deblur_image = self.decoder(FPN_info, pred_bifpn, motion_dict['motion_prior']) + input
        
        return {'deblur': [deblur_image], 'reblur' : reblur}
                



##### test  
if __name__ == '__main__':
    print('hello')