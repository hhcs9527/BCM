import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pacnet.pac import PacConv2d, PacConvTranspose2d
from DCNv2.DCN.dcn_v2 import DCN
from sub_modules.attention_zoo import *
from sub_modules.component_block import *


class Encoder_tail(nn.Module):
    def __init__(self, opt):
        super(Encoder_tail, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, 32)
        
        self.layer3 = get_norm_layer(opt.Norm, 32)

        #Conv2
        self.layer5 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, 8)

    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        #x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        return x


class Encoder_D(nn.Module):
    def __init__(self, opt):
        super(Encoder_D, self).__init__()
        '''
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        '''
        '''
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)
            )
        #Conv3
        '''
        self.layer9 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        '''
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        '''
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x


class Encoder_Light(nn.Module):
    def __init__(self, opt):
        super(Encoder_Light, self).__init__()
        #Conv1
        self.layer1 = PacConv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)

        #Conv2 
        self.layer5 = PacConv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)

        #Conv3
        self.layer9 = PacConv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        
    def forward(self, x, gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask):
        #Conv1
        x = self.layer1(x, gf_lv1)
        x = self.layer2(x) + x
        ef_lv1 = self.layer2(x) + x

        #Conv2
        x = self.layer5(ef_lv1, gf_lv2)
        x = self.layer6(x) + x
        ef_lv2 = self.layer6(x) + x

        #Conv3
        x = self.layer9(ef_lv2, gf_lv3)     
        x = self.layer10(x) + x
        x = self.layer10(x) + x 
        x = torch.cat((x, ASPP), dim = 1)

        return ef_lv1, ef_lv2, x


class Encoder_Light_PAC(nn.Module):
    def __init__(self, opt):
        super(Encoder_Light_PAC, self).__init__()
        #Conv1
        self.layer1 = PacConv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)

        #Conv2 
        self.layer5 = PacConv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)

        #Conv3
        self.layer9 = PacConv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        self.ASPP_conv = PacConv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
        
    def forward(self, x, gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask):
        #Conv1
        x = self.layer1(x, gf_lv1)
        x = self.layer2(x) + x
        ef_lv1 = self.layer2(x) + x

        #Conv2
        x = self.layer5(ef_lv1, gf_lv2)
        x = self.layer6(x) + x
        ef_lv2 = self.layer6(x) + x

        #Conv3
        x = self.layer9(ef_lv2, gf_lv3)     
        x = self.layer10(x) + x
        x = self.layer10(x) + x 
        x = self.ASPP_conv(x, ASPP)

        return ef_lv1, ef_lv2, x


class Encoder_PAC(nn.Module):
    def __init__(self, opt):
        super(Encoder_PAC, self).__init__()
        #Conv1
        self.layer1 = PacConv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)
        self.layer3 = get_norm_layer(opt.Norm, opt.channel)

        #Conv2 
        self.layer5 = PacConv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer7 = get_norm_layer(opt.Norm, opt.channel*2)

        #Conv3
        self.layer9 = PacConv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer11 = get_norm_layer(opt.Norm, opt.channel*4)
        self.ASPP_conv = PacConv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
        
    def forward(self, x, gf_lv1, gf_lv2, gf_lv3, ASPP, blur_mask):
        #Conv1
        x = self.layer1(x, gf_lv1)
        x = self.layer2(x) + x
        ef_lv1 = self.layer3(x) + x

        #Conv2
        x = self.layer5(ef_lv1, gf_lv2)
        x = self.layer6(x) + x
        ef_lv2 = self.layer7(x) + x

        #Conv3
        x = self.layer9(ef_lv2, gf_lv3)   
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        x = self.ASPP_conv(x, ASPP)

        return ef_lv1, ef_lv2, x
    

class Encoder_PAC_self(nn.Module):
    def __init__(self, opt):
        super(Encoder_PAC_self, self).__init__()
        #Conv1
        self.layer1 = PacConv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)
        self.layer3 = get_norm_layer(opt.Norm, opt.channel)

        #Conv2 
        self.layer5 = PacConv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer7 = get_norm_layer(opt.Norm, opt.channel*2)

        #Conv3
        self.layer9 = PacConv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer11 = get_norm_layer(opt.Norm, opt.channel*4)
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x, x)
        x = self.layer2(x) + x
        ef_lv1 = self.layer3(x) + x

        #Conv2
        x = self.layer5(ef_lv1, ef_lv1)
        x = self.layer6(x) + x
        ef_lv2 = self.layer7(x) + x

        #Conv3
        x = self.layer9(ef_lv2, ef_lv2)   
        x = self.layer10(x) + x
        encode_feature = self.layer11(x) + x 

        return ef_lv1, ef_lv2, encode_feature



class Encoder_pure(nn.Module):
    def __init__(self, opt):
        super(Encoder_pure, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_DCN_norm_layer(opt.Norm, opt.channel)
        self.layer3 = get_DCN_norm_layer(opt.Norm, opt.channel)

        #Conv2 
        self.layer5 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_DCN_norm_layer(opt.Norm, opt.channel*2)
        self.layer7 = get_DCN_norm_layer(opt.Norm, opt.channel*2)

        #Conv3
        self.layer9 = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        self.layer11 = get_DCN_norm_layer(opt.Norm, opt.channel*4)

        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        ef_lv1 = self.layer3(x) + x

        #Conv2
        x = self.layer5(ef_lv1)
        x = self.layer6(x) + x
        ef_lv2 = self.layer7(x) + x

        #Conv3
        x = self.layer9(ef_lv2)   
        x = self.layer10(x) + x
        x = self.layer11(x) + x 

        return ef_lv1, ef_lv2, x


class FPN_encoder_patch(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Conv1
        self.layer1 = PacConv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)
        self.layer3 = get_norm_layer(opt.Norm, opt.channel)
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.layer5 = PacConv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer7 = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv1 = nn.ConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.layer9 = PacConv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer11 = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.ConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, stride=2, padding=1, output_padding = 1)
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)

        # Conv4
        self.layer12 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv4
        self.lateral_conv4 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.ConvTranspose2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1, output_padding = 1)
        
    def forward(self, x, gf_lv1, gf_lv2, gf_lv3):
        # FPN goes up
        #Conv1
        x = self.layer1(x, gf_lv1)
        x = self.layer2(x) + x
        ef_lv1 = self.layer3(x) + x

        #Conv2
        x = self.layer5(ef_lv1, gf_lv2)
        x = self.layer6(x) + x
        ef_lv2 = self.layer7(x) + x

        #Conv3
        x = self.layer9(ef_lv2, gf_lv3)   
        x = self.layer10(x) + x
        ef_lv3 = self.layer11(x) + x 

        #Conv4
        x = self.layer12(ef_lv3)
        x = self.layer13(x) + x
        x = self.layer14(x) + x 

        # FPN top-down
        ef_lv3 = self.pred_ef_lv3(self.lateral_conv3(ef_lv3) + self.up_to_conv3(self.lateral_conv4(x)))
        ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))

        return ef_lv1, ef_lv2, ef_lv3



class FPN_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)
        self.layer3 = get_norm_layer(opt.Norm, opt.channel)
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.layer5 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer7 = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv1 = nn.ConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.layer9 = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer11 = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.ConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, stride=2, padding=1, output_padding = 1)
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)

        # Conv4
        self.layer12 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv4
        self.lateral_conv4 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.ConvTranspose2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1, output_padding = 1)
        
    def forward(self, x):
        # FPN goes up
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        ef_lv1 = self.layer3(x) + x

        #Conv2
        x = self.layer5(ef_lv1)
        x = self.layer6(x) + x
        ef_lv2 = self.layer7(x) + x

        #Conv3
        x = self.layer9(ef_lv2)   
        x = self.layer10(x) + x
        ef_lv3 = self.layer11(x) + x 

        #Conv4
        x = self.layer12(ef_lv3)
        x = self.layer13(x) + x
        x = self.layer14(x) + x 

        # FPN top-down
        ef_lv3 = self.pred_ef_lv3(self.lateral_conv3(ef_lv3) + self.up_to_conv3(self.lateral_conv4(x)))
        ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))

        return ef_lv1, ef_lv2, ef_lv3

#####
    # channel * 2
#####
class FPN_encoder_pix_shuff(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.space_to_depth = space_to_depth(2)
        # Conv1
        self.raw_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.raw_non_linear = get_norm_layer(opt.Norm, opt.channel)
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.content_lv2 = nn.Sequential(\
            nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=1, padding=1),
            self.space_to_depth, 
            nn.Conv2d(opt.channel * 2 * 4, opt.channel * 2, kernel_size=3, padding=1))

        self.content_lv2_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv1
        self.up_to_conv1 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel, kernel_size=3, padding=1))
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.content_lv3 = nn.Sequential(\
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, stride=1, padding=1),
            self.space_to_depth, 
            nn.Conv2d(opt.channel * 2 * 4, opt.channel * 2, kernel_size=3, padding=1))

        self.content_lv3_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel*2, kernel_size=3, padding=1))
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv4
        self.content_lv4 = nn.Sequential(self.space_to_depth, nn.Conv2d(opt.channel*2*4, opt.channel*2, kernel_size=3, padding=1))
        self.content_lv4_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv4
        self.lateral_conv4 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel*2, kernel_size=3, padding=1,))
        

    def forward(self, x):
        # FPN goes up
        # Conv1
        # input size (batch, 3, H, W)
        x = self.raw_layer(x)
        x = self.raw_non_linear(x) + x
        ef_lv1 = self.raw_non_linear(x) + x

        # Conv2
        # input size (batch, opt.channel, H, W)
        x = self.content_lv2(ef_lv1)
        x = self.content_lv2_non_linear(x) + x
        ef_lv2 = self.content_lv2_non_linear(x) + x

        # onv3
        # input size (batch, opt.channel * 2, H/2, W/2)
        x = self.content_lv3(ef_lv2)  
        x = self.content_lv3_non_linear(x) + x
        ef_lv3 = self.content_lv3_non_linear(x) + x 

        # Conv4
        # input size (batch, opt.channel * 2, H/4, W/4)
        x = self.content_lv4(ef_lv3)
        x = self.content_lv4_non_linear(x) + x
        x = self.content_lv4_non_linear(x) + x 

        # FPN top-down
        # input size (batch, opt.channel * 2, H/8, W/8)
        ef_lv3 = self.pred_ef_lv3(self.lateral_conv3(ef_lv3) + self.up_to_conv3(self.lateral_conv4(x)))
        ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))

        # output size (batch, opt.channel * 2, H/8, W/8) 
        # ef_lv1 (batch, opt.channel, H, W) 
        # ef_lv2 (batch, opt.channel * 2, H/2, W/2) 
        # ef_lv3 (batch, opt.channel * 2, H/4, W/4) 
        return ef_lv1, ef_lv2, ef_lv3


#####
    # channel * 4
#####

class FPN_encoder_pix_shuff(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.space_to_depth = space_to_depth(2)

        # Conv1
        self.raw_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.raw_non_linear = get_norm_layer(opt.Norm, opt.channel)
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.content_lv2 = nn.Sequential(\
            nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=1, padding=1),
            self.space_to_depth, 
            nn.Conv2d(opt.channel * 2 * 4, opt.channel * 2, kernel_size=3, padding=1))

        self.content_lv2_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv1
        self.up_to_conv1 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel, kernel_size=3, padding=1))
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.content_lv3 = nn.Sequential(\
            nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=1, padding=1),
            self.space_to_depth, 
            nn.Conv2d(opt.channel * 4 * 4, opt.channel * 4, kernel_size=3, padding=1))

        self.content_lv3_non_linear = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*4//4, opt.channel*2, kernel_size=3, padding=1))
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)

        # Conv4
        self.content_lv4 = nn.Sequential(self.space_to_depth, nn.Conv2d(opt.channel*4*4, opt.channel*4, kernel_size=3, padding=1))
        self.content_lv4_non_linear = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv4
        self.lateral_conv4 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*4//4, opt.channel*4, kernel_size=3, padding=1,))
        

    def forward(self, x):
        # FPN goes up
        # Conv1
        # input size (batch, 3, H, W)
        x = self.raw_layer(x)
        x = self.raw_non_linear(x) + x
        ef_lv1 = self.raw_non_linear(x) + x

        # Conv2
        # input size (batch, opt.channel, H, W)
        x = self.content_lv2(ef_lv1)
        x = self.content_lv2_non_linear(x) + x
        ef_lv2 = self.content_lv2_non_linear(x) + x

        # onv3
        # input size (batch, opt.channel * 2, H/2, W/2)
        x = self.content_lv3(ef_lv2)  
        x = self.content_lv3_non_linear(x) + x
        ef_lv3 = self.content_lv3_non_linear(x) + x 

        # Conv4
        # input size (batch, opt.channel * 4, H/4, W/4)
        x = self.content_lv4(ef_lv3)
        x = self.content_lv4_non_linear(x) + x
        x = self.content_lv4_non_linear(x) + x 

        # FPN top-down
        # input size (batch, opt.channel * 4, H/8, W/8)
        ef_lv3 = self.pred_ef_lv3(self.lateral_conv3(ef_lv3) + self.up_to_conv3(self.lateral_conv4(x)))
        ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))

        # output size (batch, opt.channel * 4, H/8, W/8) 
        # ef_lv1 (batch, opt.channel, H, W) 
        # ef_lv2 (batch, opt.channel * 2, H/2, W/2) 
        # ef_lv3 (batch, opt.channel * 4, H/4, W/4) 
        return ef_lv1, ef_lv2, ef_lv3


#####
    # channel * 2
#####
class FPN_encoder_pix_shuff_deep8(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.space_to_depth = space_to_depth(2)

        # Conv1
        self.raw_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.raw_non_linear = get_norm_layer(opt.Norm, opt.channel)
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.content_lv2 = nn.Sequential(\
            nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=1, padding=1),
            self.space_to_depth, 
            nn.Conv2d(opt.channel * 2 * 4, opt.channel * 2, kernel_size=3, padding=1))

        self.content_lv2_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv1
        self.up_to_conv1 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel, kernel_size=3, padding=1))
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.content_lv3 = nn.Sequential(\
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, stride=1, padding=1),
            self.space_to_depth, 
            nn.Conv2d(opt.channel * 2 * 4, opt.channel * 2, kernel_size=3, padding=1))

        self.content_lv3_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel*2, kernel_size=3, padding=1))
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv4
        self.content_lv4 = nn.Sequential(self.space_to_depth, nn.Conv2d(opt.channel*2*4, opt.channel*2, kernel_size=3, padding=1))
        self.content_lv4_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv3
        self.lateral_conv4 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel*2, kernel_size=3, padding=1))
        
        # Conv5
        self.content_lv5 = nn.Sequential(self.space_to_depth, nn.Conv2d(opt.channel*2*4, opt.channel*2, kernel_size=3, padding=1))
        self.content_lv5_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv5
        self.lateral_conv5 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv4
        self.up_to_conv4 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(opt.channel*2//4, opt.channel*2, kernel_size=3, padding=1))


    def forward(self, x):
        # FPN goes up
        # Conv1
        # input size (batch, 3, H, W)
        x = self.raw_layer(x)
        x = self.raw_non_linear(x) + x
        ef_lv1 = self.raw_non_linear(x) + x

        # Conv2
        # input size (batch, opt.channel, H, W)
        x = self.content_lv2(ef_lv1)
        x = self.content_lv2_non_linear(x) + x
        ef_lv2 = self.content_lv2_non_linear(x) + x

        # onv3
        # input size (batch, opt.channel * 2, H/2, W/2)
        x = self.content_lv3(ef_lv2)  
        x = self.content_lv3_non_linear(x) + x
        ef_lv3 = self.content_lv3_non_linear(x) + x 

        # Conv4
        # input size (batch, opt.channel * 2, H/4, W/4)
        x = self.content_lv4(ef_lv3)
        x = self.content_lv4_non_linear(x) + x
        ef_lv4 = self.content_lv4_non_linear(x) + x 

        # Conv5
        # input size (batch, opt.channel * 2, H/8, W/8)
        x = self.content_lv5(ef_lv4)
        x = self.content_lv5_non_linear(x) + x
        ef_lv5 = self.content_lv5_non_linear(x) + x 

        # FPN top-down
        # input size (batch, opt.channel * 2, H/16, W/16)
        ef_lv4 = self.lateral_conv4(ef_lv4) + self.up_to_conv4(self.lateral_conv5(ef_lv5))
        ef_lv3 = self.pred_ef_lv3(self.lateral_conv3(ef_lv3) + self.up_to_conv3(ef_lv4))
        ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))


        # output size (batch, opt.channel * 2, H/8, W/8) 
        # ef_lv1 (batch, opt.channel, H, W) 
        # ef_lv2 (batch, opt.channel * 2, H/2, W/2) 
        # ef_lv3 (batch, opt.channel * 2, H/4, W/4) 
        return ef_lv1, ef_lv2, ef_lv3


#####
    # channel * 4
#####
class FPN_encoder_deep(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Conv1
        self.raw_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.raw_non_linear = get_norm_layer(opt.Norm, opt.channel)
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.content_lv2 = nn.Sequential(\
            nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, stride=2, padding=1), 
            nn.ReLU())

        self.content_lv2_non_linear = get_norm_layer(opt.Norm, opt.channel*2)
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv1
        self.up_to_conv1 = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                            nn.Conv2d(opt.channel*2, opt.channel, kernel_size=3,padding=1),
                            nn.ReLU()) 
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.content_lv3 = nn.Sequential(\
            nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        self.content_lv3_non_linear = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                            nn.Conv2d(opt.channel*4, opt.channel*2, kernel_size=3,padding=1),
                            nn.ReLU()) 
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)

        # Conv4
        self.content_lv4 = nn.Sequential(
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1), 
            nn.ReLU())

        self.content_lv4_non_linear = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv3
        self.lateral_conv4 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3,padding=1),
                            nn.ReLU()) 
        # Conv5
        self.content_lv5 = nn.Sequential(
            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1), 
            nn.ReLU())

        self.content_lv5_non_linear = get_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv5
        self.lateral_conv5 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv4
        self.up_to_conv4 = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                            nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3,padding=1),
                            nn.ReLU()) 

    def forward(self, x):
        # FPN goes up
        # Conv1
        # input size (batch, 3, H, W)
        x = self.raw_layer(x)
        x = self.raw_non_linear(x) + x
        ef_lv1 = self.raw_non_linear(x) + x

        # Conv2
        # input size (batch, opt.channel, H, W)
        x = self.content_lv2(ef_lv1)
        x = self.content_lv2_non_linear(x) + x
        ef_lv2 = self.content_lv2_non_linear(x) + x

        # onv3
        # input size (batch, opt.channel * 2, H/2, W/2)
        x = self.content_lv3(ef_lv2)  
        x = self.content_lv3_non_linear(x) + x
        ef_lv3 = self.content_lv3_non_linear(x) + x 

        # Conv4
        # input size (batch, opt.channel * 2, H/4, W/4)
        x = self.content_lv4(ef_lv3)
        x = self.content_lv4_non_linear(x) + x
        ef_lv4 = self.content_lv4_non_linear(x) + x 

        # Conv5
        # input size (batch, opt.channel * 2, H/8, W/8)
        x = self.content_lv5(ef_lv4)
        x = self.content_lv5_non_linear(x) + x
        ef_lv5 = self.content_lv5_non_linear(x) + x 

        # FPN top-down
        # input size (batch, opt.channel * 2, H/16, W/16)
        ef_lv4 = self.lateral_conv4(ef_lv4) + self.up_to_conv4(self.lateral_conv5(ef_lv5))
        ef_lv3 = self.pred_ef_lv3(self.lateral_conv3(ef_lv3) + self.up_to_conv3(ef_lv4))
        ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))


        # output size (batch, opt.channel * 2, H/8, W/8) 
        # ef_lv1 (batch, opt.channel, H, W) 
        # ef_lv2 (batch, opt.channel * 2, H/2, W/2) 
        # ef_lv3 (batch, opt.channel * 2, H/4, W/4) 
        return ef_lv1, ef_lv2, ef_lv3



class FPN_encoder_Atrous_PAC(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.content = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.lateral_conv = nn.ModuleDict()
        self.pred_ef = nn.ModuleDict()
        self.up_conv = nn.ModuleDict()
        self.FPN_level = 4

        channel_list = [opt.channel * 2 ** (i-1) if i < 3 else opt.channel * 4 for i in range(1,self.FPN_level+1)]
        stride_list = [1 if i < 3 else 1 for i in range(1,self.FPN_level+1)]

        for i in range(1,self.FPN_level+1):
            level = f'conv{i}'
            channel = channel_list[i-1]
            if i == 1:
                self.content[level] = nn.Conv2d(3, channel, kernel_size=5, padding=2)

            elif i == 2 or i == 3:
                self.content[level] = nn.Sequential(\
                        nn.Conv2d(channel_list[i-2], channel, kernel_size=3, stride=2, padding = stride_list[i-2]),
                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=stride_list[i-1]), 
                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.LeakyReLU(0.2, True))
                self.up_conv[level] = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel_list[i-2], kernel_size=3, padding = stride_list[i-1]),
                        nn.LeakyReLU(0.2, True)) 

            else:
                self.content[level] = nn.Sequential(
                        nn.Conv2d(channel_list[i-2], channel, kernel_size=3, stride=2, padding = stride_list[i-1]),
                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=stride_list[i-1]), 
                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.LeakyReLU(0.2, True))
                self.up_conv[level] = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel_list[i-2], kernel_size=3, padding=stride_list[i-1]),
                        nn.LeakyReLU(0.2, True)) 

            #if i < self.FPN_level:
            #    self.pred_ef[level] = motion_var_SFT(channel, channel, 1)
                #PacConv2d(channel, channel, kernel_size=3, padding=stride_list[i-1])
                #self.sft_pac[deconv] = motion_var_SFT(channel_list[i-2], self.motion_channel, 1)

            self.non_linear_transform[level] = get_norm_layer(opt.Norm, channel, padding=stride_list[i-1])
            self.lateral_conv[level] = nn.Conv2d(channel, channel, kernel_size=3, padding=1)


    def forward(self, x, feature_list):
        # FPN goes up from Conv 1 - 4
        # Goes up
        ef_dict = {}
        for i in range(1,self.FPN_level+1):
            level = f'conv{i}'
            ef = f'ef_lv{i}'
            x = self.content[level](x)
            x = self.non_linear_transform[level](x) + x
            ef_dict[ef] = x

        # Goes down
        ef_dict[ef] = self.lateral_conv[level](ef_dict[ef])     
        for i in range(self.FPN_level-1, 0, -1):
            level = f'conv{i}'
            up_level = f'conv{i+1}'
            ef = f'ef_lv{i}'
            ef_last = f'ef_lv{i+1}'
            ef_dict[ef] = self.lateral_conv[level](ef_dict[ef]) + self.up_conv[up_level](ef_dict[ef_last])

        # output size (batch, opt.channel * 2, H/8, W/8) 
        # ef_lv1 (batch, opt.channel, H, W) 
        # ef_lv2 (batch, opt.channel * 2, H, W) 
        # ef_lv3 (batch, opt.channel * 4, H, W) 
        self.feature_list = []
        return ef_dict['ef_lv1'], ef_dict['ef_lv2'], ef_dict['ef_lv3']


class FPN_encoder_DCN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.content = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.lateral_conv = nn.ModuleDict()
        self.pred_ef = nn.ModuleDict()
        self.up_conv = nn.ModuleDict()
        self.FPN_level = 4

        channel_list = [opt.channel * 2 ** (i-1) if i < 3 else opt.channel * 4 for i in range(1,self.FPN_level+1)]
        stride_list = [1 if i < 3 else 1 for i in range(1,self.FPN_level+1)]

        for i in range(1,self.FPN_level+1):
            level = f'conv{i}'
            channel = channel_list[i-1]
            if i == 1:
                self.content[level] = nn.Sequential(
                        Guide_DCN_Block(in_c = 3, out_c = channel, stride_len = 1, kernel_size = 5),
                        nn.Conv2d(channel, channel, kernel_size=5, padding=2))

            elif i == 2 or i == 3:
                self.content[level] = nn.Sequential(\
                        nn.Conv2d(channel_list[i-2], channel, kernel_size=3, stride=2, padding = stride_list[i-2]),
                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=stride_list[i-1]), 
                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.LeakyReLU(0.2, True))
                self.up_conv[level] = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel_list[i-2], kernel_size=3, padding = stride_list[i-1]),
                        nn.LeakyReLU(0.2, True)) 

            else:
                self.content[level] = nn.Sequential(
                        nn.Conv2d(channel_list[i-2], channel, kernel_size=3, stride=2, padding = stride_list[i-1]),
                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=stride_list[i-1]), 
                        #RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.LeakyReLU(0.2, True))
                self.up_conv[level] = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel_list[i-2], kernel_size=3, padding=stride_list[i-1]),
                        nn.LeakyReLU(0.2, True)) 

            if i < self.FPN_level:
                self.pred_ef[level] = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

            self.non_linear_transform[level] = get_norm_layer(opt.Norm, channel, padding=stride_list[i-1])
            self.lateral_conv[level] = nn.Conv2d(channel, channel, kernel_size=3, padding=1)


    def forward(self, x):
        # FPN goes up from Conv 1 - 4
        # Goes up
        ef_dict = {}
        for i in range(1,self.FPN_level+1):
            level = f'conv{i}'
            ef = f'ef_lv{i}'
            x = self.content[level](x)
            x = self.non_linear_transform[level](x) + x
            ef_dict[ef] = x

        # Goes down
        ef_dict[ef] = self.lateral_conv[level](ef_dict[ef])     
        for i in range(self.FPN_level-1, 0, -1):
            level = f'conv{i}'
            up_level = f'conv{i+1}'
            ef = f'ef_lv{i}'
            ef_last = f'ef_lv{i+1}'
            ef_dict[ef] = self.pred_ef[level](self.lateral_conv[level](ef_dict[ef]) + self.up_conv[up_level](ef_dict[ef_last]))

        # output size (batch, opt.channel * 2, H/8, W/8) 
        # ef_lv1 (batch, opt.channel, H, W) 
        # ef_lv2 (batch, opt.channel * 2, H/2, W/2) 
        # ef_lv3 (batch, opt.channel * 4, H/4, W/4) 
        self.feature_list = [ef_dict['ef_lv1'], ef_dict['ef_lv2'], ef_dict['ef_lv3']]
        return ef_dict['ef_lv1'], ef_dict['ef_lv2'], ef_dict['ef_lv3']