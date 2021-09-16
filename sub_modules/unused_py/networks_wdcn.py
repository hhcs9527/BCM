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
from sub_modules.alignment_zoo import *
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

    elif opt.G == 'Patch_Motion_SP':
        netG = Patch_Motion_SP(opt)

    elif opt.G == 'Patch_Motion_BiFPN':
        netG = Patch_Motion_BiFPN(opt)

    elif opt.G == 'multi_scale_BiFPN':
        netG = multi_scale_BiFPN(opt)
    
    elif opt.G == 'double_reblur_DMPHN_1_2_4':
        netG = double_reblur_DMPHN_1_2_4(opt)

    elif opt.G == 'reblur_DMPHN_1_2_4':
        netG = reblur_DMPHN_1_2_4(opt)

    else:
        netG = DMPHN(opt)

    return netG.apply(weights_init).to(use_gpu)


def define_content_D(opt):
    '''
        Follow the design of Encoder in DMPHN
    '''
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.content_D == 'DMPHN':
        netD = Encoder_D(opt)
    return netD.to(use_gpu).apply(weights_init)


def define_image_D(opt):
    ''' 
        Follow the design of multi-path part of DMPHN, to get the detail part of the whole image 
    '''
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.image_D == 'multi_patch':
        netD = multi_patch_D(opt)
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

class Patch_G(nn.Module):
    def __init__(self, opt):
        super(Patch_G, self).__init__()
        self.decoder = Decoder(opt)

    def forward(self, ef_lv1, ef_lv2, encode_feature):
        return self.decoder(encode_feature, ef_lv2, ef_lv1)



class Patch_D(nn.Module):
    def __init__(self, opt):
        super(Patch_D, self).__init__()
        self.gf = Guide_Feature(opt)
        self.encoder = Encoder(opt)       
        self.encoder_tail = Encoder_tail(opt)  

    def forward(self, input):
        gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input, gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask)
        critic = self.encoder_tail(encode_feature)

        return ef_lv1, ef_lv2, encode_feature, critic
    

class Patch_ALL(nn.Module):
    def __init__(self, opt):
        super(Patch_ALL, self).__init__()
        self.gf = Guide_Feature(opt)

        if opt.ASPP_adpative:
            self.encoder = Encoder_PAC(opt) 
            self.decoder = Decoder_PAC(opt)
        else:
            self.encoder = Encoder(opt)
            self.decoder = Decoder(opt) 

    def forward(self, input):
        gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input, gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask)
        return self.decoder(encode_feature, ef_lv2, ef_lv1) + input


class Patch_Light(nn.Module):
    def __init__(self, opt):
        super(Patch_Light, self).__init__()
        self.gf = Guide_Feature(opt)

        if opt.ASPP_adpative:
            self.encoder = Encoder_Light_PAC(opt) 
            self.decoder = Decoder_Light_PAC(opt)
        else:
            self.encoder = Encoder_Light(opt) 
            self.decoder = Decoder_Light(opt)

    def forward(self, input):
        gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input, gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask)
        return self.decoder(encode_feature, ef_lv2, ef_lv1) + input


class Patch_Deblur_behind_pure(nn.Module):
    def __init__(self, opt):
        super(Patch_Deblur_behind, self).__init__()
        self.gf = Guide_Feature_Decoder(opt)
        self.encoder = Encoder_PAC_self(opt) 
        self.decoder = Decoder_PAC_self(opt)

    def forward(self, input):
        gf_lv1, gf_lv2, gf_lv3 = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input)
        return self.decoder(encode_feature, ef_lv2, ef_lv1, gf_lv1, gf_lv2, gf_lv3) + input


class Patch_Deblur_behind(nn.Module):
    def __init__(self, opt):
        super(Patch_Deblur_behind, self).__init__()
        self.gf = Guide_Feature_Decoder(opt)
        self.encoder = Encoder_pure(opt) 
        self.decoder = Decoder_PAC_self(opt)

    def forward(self, input):
        gf_lv1, gf_lv2, gf_lv3 = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input)
        return self.decoder(encode_feature, ef_lv2, ef_lv1, gf_lv1, gf_lv2, gf_lv3) + input


class Patch_Deblur_FPN_patch(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_Feature_Decoder(opt)
        self.encoder = FPN_encoder_patch(opt) 
        self.decoder = Decoder_PAC(opt)

    def forward(self, input):
        gf_lv1, gf_lv2, gf_lv3 = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input, gf_lv1, gf_lv2, gf_lv3)
        return self.decoder(encode_feature, ef_lv2, ef_lv1) + input


class Patch_FPN_VAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Feature_Distribution(opt)
        self.encoder = FPN_encoder(opt) 
        self.decoder = Decoder_FPN(opt)

    def forward(self, input):
        blur_feature, sharp_feature = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input)
        return {'deblur': self.decoder(encode_feature, ef_lv2, ef_lv1, sharp_feature), 
                'blur': self.decoder(encode_feature, ef_lv2, ef_lv1, blur_feature)}



class Patch_FPN_DCN(nn.Module):
    '''
        when consider the offset, also the transformation on latent space (e.g. including convlstm, alignment)
        we execute those operations on motion latent feature.
    '''
    def __init__(self, opt):
        super().__init__()
        self.gf = Motion_DCN_LSTM(opt)
        self.encoder = FPN_encoder_DCN(opt) 
        self.decoder = FPN_Decoder_DCN(opt)
        self.recurrent_time = opt.Recurrent_times
        self.ConvLSTM = ConvLSTM(input_dim = opt.channel*4, hidden_dim=opt.channel*4, kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)


    def generate_deblur(self, ef_lv1, ef_lv2, content_feature, input_, motion_feature):
        '''
            generate image with motion and encode feature.
        '''
        return self.decoder(content_feature, ef_lv2, ef_lv1, motion_feature) + input_


    def generate_motion(self, motion_feature):
        '''
            use self.recurrent_time - 1 times of self.ConvLSTM
            return :
                self.motion_feature_list : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]

        '''
        self.motion_feature_list = [motion_feature]
        hidden_state = None
        for i in range(self.recurrent_time - 1):
            motion = self.motion_feature_list[i]
            layer_outputs, hidden_state = self.ConvLSTM(torch.unsqueeze(self.motion_feature_list[i], 1), hidden_state)
            self.motion_feature_list.append(torch.squeeze(layer_outputs[0], 1))


    
    def generate_all(self, ef_lv1, ef_lv2, content_feature, input_, motion_feature):
        '''
            1. generate time-series -> align feature
            2. generate the deblur image for each stage
        '''
        self.reproduce_blur = []
        self.generate_motion(motion_feature)

        for motion in self.motion_feature_list:
        #motion = torch.cat(self.motion_feature_list, dim = 1)
            self.reproduce_blur.append(self.generate_deblur(ef_lv1, ef_lv2, content_feature, input_, motion))


    def forward(self, input):
        '''
            return 
                1.  content_feature : encode result of encoder
                2.  alignment_deblur : self.alignment_deblur
                3.  reproduce_blur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
                4.  offset : self.total_offset
        '''
        motion_feature = self.gf(input)
        ef_lv1, ef_lv2, content_feature = self.encoder(input)
        self.generate_all(ef_lv1, ef_lv2, content_feature, input, motion_feature)
        return {'content_feature': content_feature,  'reproduce_blur':self.reproduce_blur}



class Reblur_Net_(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DCN = nn.ModuleDict()
        self.kernel = 3
        self.recurrent = opt.Recurrent_times - 1
        for i in range(self.recurrent):
            recurrent_lv = f'recurrent_lv{i+1}'
            self.DCN[recurrent_lv] = DCNv2(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, dilation = 1)

    def forward(self, sharp, motion):
        reblur_list = [sharp]
        for i in range(self.recurrent):
            recurrent_lv = f'recurrent_lv{i+1}'
            offset, weight = torch.split(motion[0], [self.kernel*self.kernel*2, self.kernel*self.kernel], dim=1)
            reblur_img = self.DCN[recurrent_lv](reblur_list[i], offset, weight)
            reblur_list.append(reblur_img)

        self.feature_list = reblur_list

        return sum(self.feature_list)/(self.recurrent+1)


class Reblur_Net(nn.Module):
    def __init__(self, opt):
        super().__init__()
        opt.channel = opt.channel//2
        self.DCN = nn.ModuleDict()
        self.kernel = 3
        self.recurrent = opt.Recurrent_times - 1

        self.encoder = Encoder(opt)
        self.decoder = Decoder_motion_DCN(opt)

    def forward(self, sharp, motion):
        reblur_list = [sharp]
        for i in range(self.recurrent):
            sharp_feature = self.encoder(sharp)
            reblur_img = self.decoder(sharp_feature, motion[i])
            reblur_list.append(reblur_img + sharp)

        self.feature_list = reblur_list

        return sum(self.feature_list)/(self.recurrent+1)


class Patch_Motion_SFT_fusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Sharp_Distribution(opt)
        self.encoder = FPN_encoder(opt) 
        self.decoder = FPN_Decoder_SFT_fusion(opt)
        self.reblur_net = Reblur_Net(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input)
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_ow'])
        
        return {'deblur': [self.decoder(encode_feature, ef_lv2, ef_lv1, motion_dict['motion_prior'], input) + input], 'reblur' : reblur,
                'motion_ow' : motion_dict['motion_ow']}



class Patch_Motion_pe0r_pix(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_pixel_offset(opt)
        self.encoder = FPN_encoder(opt) 
        self.decoder = FPN_Decoder_SFT_fusion(opt)
        self.reblur_net = motion_DF()
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        ef_lv1, ef_lv2, encode_feature = self.encoder(input)
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        return {'deblur': [self.decoder(encode_feature, ef_lv2, ef_lv1, motion_dict['motion_prior'], input) + input], 'reblur' : reblur}


class Patch_Motion_per_pix(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_pixel_offset(opt)
        self.encoder = PAN_encoder(opt) 
        self.decoder = FPN_Decoder_SFT_fusion(opt)
        self.reblur_net = motion_DF(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        encode_feature = self.encoder(input)
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        motion_info = torch.cat((motion_dict['motion_prior']), dim = 1)
        
        return {'deblur': [self.decoder(encode_feature, motion_info) + input], 'reblur' : reblur}


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
        motion_info = torch.cat((motion_dict['motion_prior']), dim = 1)

        FPN_info, pred_bifpn = self.encoder(input, motion_info)
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        
        deblur_image = self.decoder(FPN_info, pred_bifpn, motion_info) + input
        
        return {'deblur': [deblur_image], 'reblur' : reblur}


class Patch_Motion_BiFPN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_pixel_offset(opt) #Guide_SP_motion(opt)
        self.encoder = BiFPN_Encoder(opt) 
        self.decoder = BiFPN_Decoder(opt) #FPN_Decoder_SFT(opt)
        self.reblur_net = motion_DF(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        motion_info = torch.cat((motion_dict['motion_prior']), dim = 1)

        FPN_info, pred_bifpn = self.encoder(input, motion_info)

        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        #deblur_image = self.decoder(pred_bifpn, motion_info) + input
        deblur_image = self.decoder(FPN_info, pred_bifpn, motion_info) + input

        return {'deblur': [deblur_image], 'reblur' : reblur}

'''
class multi_scale_BiFPN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_pixel_offset(opt) #Guide_SP_motion(opt)
        self.encoder = multi_scale_BiFPN_Encoder(opt) 
        self.decoder = FPN_Decoder_SFT(opt) #BiFPN_Decoder(opt) 
        self.reblur_net = motion_DF(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        motion_info = torch.cat((motion_dict['motion_prior']), dim = 1)

        pred_bifpn = self.encoder(input, motion_info)

        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        #deblur_image = self.decoder(pred_bifpn, motion_info) + input
        deblur_image = self.decoder(pred_bifpn, motion_info) + input

        return {'deblur': [deblur_image], 'reblur' : reblur}
'''



class multi_scale_BiFPN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Patch_Encoder(opt)
        self.decoder = FPN_Decoder_SFT(opt) 
        self.reblur_net = motion_DF(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        deblur_dict = self.encoder(input)
        motion_info = torch.cat((deblur_dict['motion_prior']), dim = 1)

        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, deblur_dict['motion_prior'])
        
        #deblur_image = self.decoder(pred_bifpn, motion_info) + input
        deblur_image = self.decoder(deblur_dict['content_feature'], motion_info) + input

        return {'deblur': [deblur_image], 'reblur' : reblur}


class PANet_Motion_per_pix(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.gf = Guide_pixel_offset(opt)
        self.encoder = PAN_group_encoder(opt) 
        self.decoder = PAN_Decoder(opt)
        self.reblur_net = motion_DF(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        motion_dict = self.gf(input)
        encode_feature = self.encoder(input)
        
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, motion_dict['motion_prior'])
        
        motion_info = torch.cat((motion_dict['motion_prior']), dim = 1)

        return {'deblur': [self.decoder(encode_feature['encode_feature'], motion_info) + input], 'reblur' : reblur}


class PANet_self_reblur(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Patch_Unet_Encoder(opt) 
        self.decoder = Patch_Unet_Decoder(opt)
        self.reblur_net = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        encode_dict = self.encoder(input)

        decode_dict = self.decoder(encode_dict['encode_feature'])
        deblur = [decode_dict['deblur'] + input]


        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, decode_dict['reblur_filter'] + encode_dict['reblur_filter'])


        return {'deblur': deblur, 'reblur' : reblur}


class PANet_self_reblur_1_2_4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Patch_Unet_Encoder_1_2_4(opt) 
        self.decoder = Patch_Unet_Decoder_1_2_4(opt)
        self.reblur_net = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        encode_dict = self.encoder(input)

        decode_dict = self.decoder(encode_dict['encode_feature'])
        deblur = [decode_dict['deblur'] + input]


        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, decode_dict['reblur_filter'] + encode_dict['reblur_filter'])


        return {'deblur': deblur, 'reblur' : reblur}


class PANet_self_reblur_1_2_4_8(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Patch_Unet_Encoder_1_2_4_8(opt) 
        self.decoder = Patch_Unet_Decoder_1_2_4_8(opt)
        self.reblur_net = blur_understanding_module(opt)
        self.opt = opt

    def forward(self, input, sharp):
        self.istrain = self.opt.isTrain
        encode_dict = self.encoder(input)

        decode_dict = self.decoder(encode_dict['encode_feature'])
        deblur = [decode_dict['deblur'] + input]
        
        
        reblur = None
        if self.istrain:
            reblur = self.reblur_net(sharp, decode_dict['reblur_filter'] + encode_dict['reblur_filter'])


        return {'deblur': deblur, 'reblur' : reblur}


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


class DMPHN(nn.Module):
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

if __name__ == '__main__':
    print('hello')
