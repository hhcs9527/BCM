import torch
import torch.nn as nn
import torch.nn.functional as F

from pacnet.pac import PacConv2d, PacConvTranspose2d
from DCNv2.dcn_v2 import DCN
from sub_modules.component_block import *
from sub_modules.attention_zoo import *

class Decoder_Light(nn.Module):
    def __init__(self, opt):
        super(Decoder_Light, self).__init__()        
        # Deconv3
        self.layer13 = get_norm_down_layer(opt.Norm, opt.channel*4*2)
        self.middle_layer13 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1, padding=0)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_ef_lv1_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)
        
        #Deconv2
        self.layer17 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)
        
        #Deconv1
        self.layer21 = get_norm_layer(opt.Norm, opt.channel)
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self,x, ef_lv2, ef_lv1):
        #Deconv3
        x = self.layer13(x)
        x = self.middle_layer13(x) + x
        x = self.layer16(x, self.adjust_ef_lv2_pacconv(ef_lv2))
             
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer17(x) + x
        x = self.layer20(x, self.adjust_ef_lv1_pacconv(ef_lv1))

        #Deconv1
        x = self.layer21(x) + x
        x = self.layer21(x) + x
        x = self.layer24(x)

        return x

# fit the same length of original one, use 1*1 conv replace
class Decoder_Light_PAC(nn.Module):
    def __init__(self, opt):
        super(Decoder_Light_PAC, self).__init__()        
        # Deconv3
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.middle_layer13 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1, padding=0)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_ef_lv1_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)
        
        #Deconv2
        self.layer17 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)
        
        #Deconv1
        self.layer21 = get_norm_layer(opt.Norm, opt.channel)
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self,x, ef_lv2, ef_lv1):
        #Deconv3
        x = self.layer13(x) + x
        x = self.middle_layer13(x) + x
        x = self.layer16(x, self.adjust_ef_lv2_pacconv(ef_lv2))
             
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer17(x) + x
        x = self.layer20(x, self.adjust_ef_lv1_pacconv(ef_lv1))

        #Deconv1
        x = self.layer21(x) + x
        x = self.layer21(x) + x
        x = self.layer24(x)

        return x

class Decoder_PAC(nn.Module):
    def __init__(self, opt):
        super(Decoder_PAC, self).__init__()        
        # Deconv3
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_ef_lv1_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv2
        self.layer17 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer18 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv1
        self.layer21 = get_norm_layer(opt.Norm, opt.channel)
        self.layer22 = get_norm_layer(opt.Norm, opt.channel)
        self.layer24 = nn.Sequential(nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1))

        
    def forward(self,x, ef_lv2, ef_lv1):
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x, self.adjust_ef_lv2_pacconv(ef_lv2))
             
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x, self.adjust_ef_lv1_pacconv(ef_lv1))

        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)

        return x


class Decoder_PAC_self(nn.Module):
    def __init__(self, opt):
        super(Decoder_PAC_self, self).__init__()        
        # Deconv3
        self.layer13 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        self.adjust_ef_lv3_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_gf_lv3_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1, stride = 2)
        self.deblur_module_lv3 = PacConv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv2
        self.layer17 = get_DCN_norm_layer(opt.Norm, opt.channel*2)
        self.layer18 = get_DCN_norm_layer(opt.Norm, opt.channel*2)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.adjust_gf_lv2_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1, stride = 2)
        self.deblur_module_lv2 = PacConv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv1
        self.layer21 = get_DCN_norm_layer(opt.Norm, opt.channel)
        self.layer22 = get_DCN_norm_layer(opt.Norm, opt.channel)
        self.deblur_module_lv1 = PacConv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # ASPP aggregagate
        self.transform_layer_1 = nn.ConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.transform_layer_2 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
        #nn.ConvTranspose2d(opt.channel, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.transform_layer_3 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # ASPP shuffle
        self.ASPP_aggregate = nn.Sequential(
            nn.Conv2d(opt.channel*3, opt.channel, kernel_size=3, padding=1),
            nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1))       

        
    def forward(self,x, ef_lv2, ef_lv1, gf_lv1, gf_lv2, gf_lv3):
        # Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        # upscale with correspinding clear feature
        # x size: [batch, opt.channel*4, 64, 64]
        # self.adjust_gf_lv3_pacconv(gf_lv3) size: [batch, opt.channel*4, 64, 64]
        x = self.deblur_module_lv3(x, self.adjust_gf_lv3_pacconv(gf_lv3))
        # self.adjust_ef_lv3_pacconv(ef_lv2) size: [batch, opt.channel*4, 128, 128]
        de_lv1 = self.layer16(x, self.adjust_ef_lv3_pacconv(ef_lv2))

        # Deconv2
        x = self.layer17(de_lv1) + de_lv1
        x = self.layer18(x) + x
        # upscale with correspinding clear feature
        # x size: [batch, opt.channel*2, 128, 128]
        # self.adjust_gf_lv3_pacconv(gf_lv3) size: [batch, opt.channel*2, 128, 128]
        x = self.deblur_module_lv2(x, self.adjust_gf_lv2_pacconv(gf_lv2))
        # self.adjust_ef_lv3_pacconv(ef_lv2) size: [batch, opt.channel*2, 256, 256]
        de_lv2 = self.layer20(x, self.adjust_ef_lv2_pacconv(ef_lv1))

        # havent done
        # Deconv1
        x = self.layer21(de_lv2) + de_lv2
        x = self.layer22(x) + x
        # x size: [batch, opt.channel, 256, 256]
        # gf_lv1 size: [batch, opt.channel, 256, 256]
        de_lv3 = self.deblur_module_lv1(x, gf_lv1)

        # ASPP agg
        # each size after transform[batch, opt.channel, 256, 256]
        feature = torch.cat((self.transform_layer_1(de_lv1),
                            self.transform_layer_2(de_lv2),
                            self.transform_layer_3(de_lv3)),
                            dim = 1) 

        return self.ASPP_aggregate(feature)




class Decoder_PAC_FPN(nn.Module):
    def __init__(self, opt):
        super(Decoder_PAC_self, self).__init__()        
        # Deconv3
        self.layer13 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        self.adjust_ef_lv3_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_gf_lv3_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1, stride = 2)
        self.deblur_module_lv3 = PacConv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv2
        self.layer17 = get_DCN_norm_layer(opt.Norm, opt.channel*2)
        self.layer18 = get_DCN_norm_layer(opt.Norm, opt.channel*2)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.adjust_gf_lv2_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1, stride = 2)
        self.deblur_module_lv2 = PacConv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv1
        self.layer21 = get_DCN_norm_layer(opt.Norm, opt.channel)
        self.layer22 = get_DCN_norm_layer(opt.Norm, opt.channel)
        self.deblur_module_lv1 = PacConv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # ASPP aggregagate
        self.transform_layer_1 = nn.ConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.transform_layer_2 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
        #nn.ConvTranspose2d(opt.channel, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.transform_layer_3 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # ASPP shuffle
        self.ASPP_aggregate = nn.Sequential(
            nn.Conv2d(opt.channel*3, opt.channel, kernel_size=3, padding=1),
            nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1))       

        
    def forward(self,x, ef_lv2, ef_lv1, gf_lv1, gf_lv2, gf_lv3):
        # Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        # upscale with correspinding clear feature
        # x size: [batch, opt.channel*4, 64, 64]
        # self.adjust_gf_lv3_pacconv(gf_lv3) size: [batch, opt.channel*4, 64, 64]
        x = self.deblur_module_lv3(x, self.adjust_gf_lv3_pacconv(gf_lv3))
        # self.adjust_ef_lv3_pacconv(ef_lv2) size: [batch, opt.channel*4, 128, 128]
        de_lv1 = self.layer16(x, self.adjust_ef_lv3_pacconv(ef_lv2))

        # Deconv2
        x = self.layer17(de_lv1) + de_lv1
        x = self.layer18(x) + x
        # upscale with correspinding clear feature
        # x size: [batch, opt.channel*2, 128, 128]
        # self.adjust_gf_lv3_pacconv(gf_lv3) size: [batch, opt.channel*2, 128, 128]
        x = self.deblur_module_lv2(x, self.adjust_gf_lv2_pacconv(gf_lv2))
        # self.adjust_ef_lv3_pacconv(ef_lv2) size: [batch, opt.channel*2, 256, 256]
        de_lv2 = self.layer20(x, self.adjust_ef_lv2_pacconv(ef_lv1))

        # havent done
        # Deconv1
        x = self.layer21(de_lv2) + de_lv2
        x = self.layer22(x) + x
        # x size: [batch, opt.channel, 256, 256]
        # gf_lv1 size: [batch, opt.channel, 256, 256]
        de_lv3 = self.deblur_module_lv1(x, gf_lv1)

        # ASPP agg
        # each size after transform[batch, opt.channel, 256, 256]
        feature = torch.cat((self.transform_layer_1(de_lv1),
                            self.transform_layer_2(de_lv2),
                            self.transform_layer_3(de_lv3)),
                            dim = 1) 

        return self.ASPP_aggregate(feature)


class Decoder_FPN(nn.Module):
    def __init__(self, opt):
        super().__init__()    
        # Deconv3
        self.layer13 = get_norm_down_layer(opt.Norm, opt.channel*4*2)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_ef_lv1_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.layer16 = PacConv2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)
        self.attention_ef_lv2 = eca_layer(channel = opt.channel*4, k_size = 3)

        #Deconv2
        self.layer17 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer18 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer20 = PacConv2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)
        self.attention_ef_lv1 = eca_layer(channel = opt.channel*2, k_size = 3)

        #Deconv1
        self.layer21 = get_norm_layer(opt.Norm, opt.channel)
        self.layer22 = get_norm_layer(opt.Norm, opt.channel)
        self.layer24 = nn.Sequential(nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1))

        
    def forward(self,x, ef_lv2, ef_lv1, sharp_feature):
        #Deconv3
        x = torch.cat((x, sharp_feature), dim = 1)
        x = self.layer13(x) 
        x = self.layer14(x) + x
        x = self.layer16(x, self.attention_ef_lv2(self.adjust_ef_lv2_pacconv(ef_lv2)))
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x, self.attention_ef_lv2(self.adjust_ef_lv1_pacconv(ef_lv1)))

        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)

        return x


class FPN_Decoder_SFT_fusionioioi(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.SFT_up_Dconv3 = SFTLayer(opt.channel*4)
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_ef_lv1_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)
        self.attention_ef_lv2 = eca_layer(channel = opt.channel*4, k_size = 3)
        

        #Deconv2
        self.layer17 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer18 = get_norm_layer(opt.Norm, opt.channel*2)
        #self.SFT_up_Dconv2 = SFTLayer(opt.channel*2)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)
        self.attention_ef_lv1 = eca_layer(channel = opt.channel*2, k_size = 3)

        #Deconv1
        self.layer21 = get_norm_layer(opt.Norm, opt.channel)
        self.layer22 = get_norm_layer(opt.Norm, opt.channel)
        #self.SFT_up_Dconv1 = SFTLayer(opt.channel)
        self.layer24 = nn.Sequential(nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1))



        
    def forward(self,x, ef_lv2, ef_lv1, sharp_feature):
        ####
        # input size : (batch, channel * 4, H/4, W/4)
        ####
        #Deconv3
        x = self.SFT_up_Dconv3(x, sharp_feature)
        x = self.layer13(x) 
        x = self.layer14(x) + x
        x = self.layer16(x, self.attention_ef_lv2(self.adjust_ef_lv2_pacconv(ef_lv2)))
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x, self.attention_ef_lv2(self.adjust_ef_lv1_pacconv(ef_lv1)))

        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)

        return x


class Decoder_PAC_FPN(nn.Module):
    def __init__(self, opt):
        super().__init__()       
        self.PAC_fuse_layer = PacConv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=1, padding=1) 
        # Deconv3
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        self.adjust_ef_lv2_pacconv = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, padding=1)
        self.adjust_ef_lv1_pacconv = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, padding=1)
        self.layer16 = PacConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv2
        self.layer17 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer18 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer20 = PacConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, output_padding = 1, stride=2, padding=1)

        #Deconv1
        self.layer21 = get_norm_layer(opt.Norm, opt.channel)
        self.layer22 = get_norm_layer(opt.Norm, opt.channel)
        self.layer24 = nn.Sequential(nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1))

        
    def forward(self,x, ef_lv2, ef_lv1, sharp_feature):

        x = self.PAC_fuse_layer(x, sharp_feature)
        
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x, self.adjust_ef_lv2_pacconv(ef_lv2))
             
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x, self.adjust_ef_lv1_pacconv(ef_lv1))

        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

# channel * 4
class FPN_Decoder_SFT_fusion_pix_shuff_(nn.Module):
    #####
        # we reduce the whole channel_size to channel * 2 to reduce parameter,
        # also resusing the motion feature without changing it.
        # Let it 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.Deconv3_sft = motion_SFTLayer(opt.channel*4)
        self.Deconv3_non_linear_transform = get_norm_layer(opt.Norm, opt.channel*4)
        self.Deconv3_conv_up = nn.Sequential(
            eca_layer(channel = opt.channel * 4, k_size = 3), 
            nn.PixelShuffle(2))
        self.Deconv3_pac = PacConv2d(opt.channel * 4 // 4, opt.channel*2, kernel_size=3, stride=1, padding=1)
        
        #Deconv2
        self.Deconv2_sft = motion_SFTLayer(opt.channel*2)
        self.Deconv2_non_linear_transform = get_norm_layer(opt.Norm, opt.channel*2)
        self.Deconv2_conv_up = nn.Sequential(
            eca_layer(channel = opt.channel * 2, k_size = 3), 
            nn.PixelShuffle(2))
        self.Deconv2_pac = PacConv2d(opt.channel*2 // 4, opt.channel, kernel_size=3, stride=1, padding=1)

        #Deconv1
        self.Deconv1_sft = motion_SFTLayer(opt.channel)
        self.Deconv1_non_linear_transform = get_norm_layer(opt.Norm, opt.channel)
        self.Decoder_out = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(opt.channel, 3, kernel_size = 3, padding=1))
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature):
        ####
        # input encoder_feature (x) size : (batch, channel * 2, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel * 2, H, W)
        # motion feature size : (batch, channel * 2, H/4, W/4)
        ####
        #Deconv3
        # input size : (batch, channel * 2, H/4, W/4)
        residual_feature = x
        for _ in range(2):
            x = self.SFT_fusion(x, motion_feature, self.Deconv3_non_linear_transform, self.Deconv3_sft)
        x = self.Deconv3_conv_up(x + residual_feature)
        x = self.Deconv3_pac(x, ef_lv2)

        #Deconv2
        # input size : (batch, channel * 2, H/2, W/2)
        residual_feature = x
        for _ in range(2):
            x = self.SFT_fusion(x, motion_feature, self.Deconv2_non_linear_transform, self.Deconv2_sft)
        x = self.Deconv2_conv_up(x + residual_feature)
        x = self.Deconv2_pac(x, ef_lv1)

        #Deconv1
        # input size : (batch, channel * 2, H, W)
        residual_feature = x
        for _ in range(2):
            x = self.SFT_fusion(x, motion_feature, self.Deconv1_non_linear_transform, self.Deconv1_sft)

        return self.Decoder_out(x + residual_feature)

# channel * 2
class FPN_Decoder_SFT_fusion_pix_shuffjlk(nn.Module):
    #####
        # we reduce the whole channel_size to channel * 2 to reduce parameter,
        # also resusing the motion feature without changing it.
        # Let it 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.offset_transpose_conv = nn.ModuleDict()
        self.feat_transpose_conv = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.dcn = nn.ModuleDict()
        self.gen_ow = nn.ModuleDict()  
        
        # Deconv3
        self.Deconv3_sft = motion_SFTLayer(opt.channel*2)
        self.Deconv3_non_linear_transform = get_norm_layer(opt.Norm, opt.channel*2)
        self.Deconv3_conv_up = nn.Sequential(
            eca_layer(channel = opt.channel * 2, k_size = 3), 
            nn.PixelShuffle(2))
        self.Deconv3_pac = PacConv2d(opt.channel * 2 // 4, opt.channel*2, kernel_size=3, stride=1, padding=1)
        
        #Deconv2
        self.Deconv2_sft = motion_SFTLayer(opt.channel*2)
        self.Deconv2_non_linear_transform = get_norm_layer(opt.Norm, opt.channel*2)
        self.Deconv2_conv_up = nn.Sequential(
            eca_layer(channel = opt.channel * 2, k_size = 3), 
            nn.PixelShuffle(2))
        self.Deconv2_pac = PacConv2d(opt.channel*2 // 4, opt.channel, kernel_size=3, stride=1, padding=1)

        #Deconv1
        self.Deconv1_sft = motion_SFTLayer(opt.channel)
        self.Deconv1_non_linear_transform = get_norm_layer(opt.Norm, opt.channel)
        self.Decoder_out = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(opt.channel, 3, kernel_size = 3, padding=1))
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature):
        ####
        # input encoder_feature (x) size : (batch, channel * 2, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel * 2, H, W)
        # motion feature size : (batch, channel * 2, H/4, W/4)
        ####
        # Deconv3
        # input size : (batch, channel * 2, H/4, W/4)
        residual_feature = x
        for _ in range(2):
            x = self.SFT_fusion(x, motion_feature, self.Deconv3_non_linear_transform, self.Deconv3_sft)
        x = self.Deconv3_conv_up(x + residual_feature)
        x = self.Deconv3_pac(x, ef_lv2)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)
        residual_feature = x
        for _ in range(2):
            x = self.SFT_fusion(x, motion_feature, self.Deconv2_non_linear_transform, self.Deconv2_sft)
        x = self.Deconv2_conv_up(x + residual_feature)
        x = self.Deconv2_pac(x, ef_lv1)

        # Deconv1
        # input size : (batch, channel * 2, H, W)
        residual_feature = x
        for _ in range(2):
            x = self.SFT_fusion(x, motion_feature, self.Deconv1_non_linear_transform, self.Deconv1_sft)

        return self.Decoder_out(x + residual_feature)



# channel * 2
class FPN_Decoder_SFT_fusion_pix_shuf798f(nn.Module):
    #####
        # we reduce the whole channel_size to channel * 2 to reduce parameter,
        # also resusing the motion feature without changing it.
        # Let it 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.motion_channel = opt.channel*2

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = opt.channel
                self.conv_up[deconv] = nn.Sequential(
                                        eca_layer(channel = channel, k_size = 3),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1), 
                                        nn.ReLU())
            else:
                channel = opt.channel*2
                self.sft_pac[deconv] = motion_SFTLayer(channel)
                self.conv_up[deconv] = nn.Sequential(
                        eca_layer(channel = channel, k_size = 3), 
                        nn.PixelShuffle(2))

            if i == 3:
                self.pac[deconv] = PacConv2d(channel // 4, channel, kernel_size=3, stride=1, padding=1)
            elif i == 2:
                self.pac[deconv] = PacConv2d(channel // 4, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = PacConv2d(opt.channel, 3, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = motion_SFT(channel, self.motion_channel)
            self.sft_lv2[deconv] = motion_SFT(channel, self.motion_channel)
            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 2, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 2, H/4, W/4)
        ####
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x.clone()
            else:
                x = pac_result
                residual_feature = x.clone()
            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])
            styled_feature = self.conv_up[deconv](x + residual_feature)
            if i == 3:
                pac_result = self.pac[deconv](styled_feature, ef_lv2)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)
            elif i == 2:
                pac_result = self.pac[deconv](styled_feature, ef_lv1)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)
            else:
                pac_result = self.pac[deconv](styled_feature, input)

        # Deconv3
        # input size : (batch, channel * 2, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result



# 1-2-4
class FPN_Decoder_SFT_fusion_pix_shuff_diff(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.motion_channel = opt.channel*4

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = opt.channel
                self.conv_up[deconv] = nn.Sequential(
                                        eca_layer(channel = channel, k_size = 3),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1), 
                                        nn.ReLU())
            else:
                channel = opt.channel*2*(i-1)
                
                self.conv_up[deconv] = nn.Sequential(
                        eca_layer(channel = channel, k_size = 3), 
                        nn.Conv2d(channel, channel*4, kernel_size = 3, padding=1),
                        nn.PixelShuffle(2))
                self.sft_pac[deconv] = motion_var_SFT(channel//2, self.motion_channel, 2*(3 - i + 1))

            if i != 1:
                self.pac[deconv] = PacConv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = motion_var_SFT(channel, self.motion_channel, 2*(3 - i))
            self.sft_lv2[deconv] = motion_var_SFT(channel, self.motion_channel, 2*(3 - i))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x.clone()
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x.clone()
            else:
                x = pac_result
                residual_feature = x.clone()

            x = self.SFT_fusion(x, motion_feature[i-1], self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            x = self.SFT_fusion(x, motion_feature[i-1], self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])
            styled_feature = self.conv_up[deconv](x + residual_feature)
            if i == 3:
                pac_result = self.pac[deconv](styled_feature, ef_lv2)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature[i-1])
            elif i == 2:
                pac_result = self.pac[deconv](styled_feature, ef_lv1)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature[i-1])

            else:
                pac_result = self.pac[deconv](styled_feature)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result
    

    # 1-2-4
class FPN_Decoder_SFT_fusion(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.motion_channel = 3 ** 3 #opt.channel
        
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = opt.channel
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = opt.channel*2*(i-1)
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))
                
                #self.sft_pac[deconv] = motion_var_SFT(channel//2, self.motion_channel, 2*(3 - i + 1))

            if i != 1:
                self.pac[deconv] = PacConv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = SFT_DCN(channel, self.motion_channel, 2**(3-i))
            #self.sft_lv2[deconv] = SFT_DCN(channel, self.motion_channel, 2**(3-i))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            #self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.feature_list = []
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = pac_result
                residual_feature = x
    
            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            #x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])

            styled_feature = self.conv_up[deconv](x + residual_feature)
            if i == 3:
                pac_result = self.pac[deconv](styled_feature, ef_lv2)
                #pac_result = self.sft_pac[deconv](pac_result, motion_feature)
                
            elif i == 2:
                pac_result = self.pac[deconv](styled_feature, ef_lv1)
                #pac_result = self.sft_pac[deconv](pac_result, motion_feature)

            else:
                pac_result = self.pac[deconv](styled_feature)
            
            self.feature_list.append(pac_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result


    # 1-2-4
class FPN_Decoder_SFT_Atrous(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.ref = nn.ModuleDict()
        self.motion_channel = opt.channel*4

        channel_list = [opt.channel * 2**(i) if i < 3 else opt.channel*4 for i in range(3)]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=i, dilation=i),
                                        nn.LeakyReLU(0.2, True))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=i, dilation=i),
                        nn.LeakyReLU(0.2, True))
                
                self.sft_pac[deconv] = motion_var_SFT(channel_list[i-2], self.motion_channel, 1)

            if i != 1:
                self.pac[deconv] = nn.Conv2d(channel, channel_list[i-2], kernel_size=3, stride=1, padding=i, dilation=i)
               # self.pac[deconv] = PacConv2d(channel, channel_list[i-2], kernel_size=3, stride=1, padding=1)
            else:
                #self.pac[deconv] = PacConv2d(opt.channel, 3, kernel_size=3, stride=1, padding=1)
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=i, dilation=i)
                #self.ref[deconv] = PacConv2d(3, 3, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = motion_var_SFT(channel, self.motion_channel, 1)
            #self.sft_lv2[deconv] = motion_var_SFT(channel, self.motion_channel, 1)

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel, padding=i, dilation=i)
            #self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel, padding=i, dilation=i)
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x.clone()
            else:
                x = pac_result
                residual_feature = x.clone()

            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            #x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])

            styled_feature = self.conv_up[deconv](x + residual_feature)

            if i == 3:
                #print(styled_feature.size(), ef_lv2.size())
                pac_result = self.pac[deconv](styled_feature)#, ef_lv2)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)

            elif i == 2:
                pac_result = self.pac[deconv](styled_feature)#, ef_lv1)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)

            else:
                #pac_result = self.pac[deconv](styled_feature, input)
                pac_result = self.pac[deconv](styled_feature)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result + input
        #self.ref[deconv](pac_result, input)



class FPN_Decoder_SFT_Atrous_Unet(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.adjust = nn.ModuleDict()
        #self.motion_channel = 3

        channel_list = [opt.channel * 2**(i) if i < 3 else opt.channel*4 for i in range(3)]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1, dilation=1),
                                        nn.LeakyReLU(0.2, True))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1, dilation=1),
                        nn.LeakyReLU(0.2, True))

            if i != 1:
                self.pac[deconv] = nn.Sequential(get_norm_layer(opt.Norm, channel, padding=1, dilation=1),
                                     nn.Conv2d(channel, channel_list[i-2], kernel_size=3, stride=1, padding=1, dilation=1))
            else:
                self.pac[deconv] = nn.Sequential(get_norm_layer(opt.Norm, channel, padding=1, dilation=1),
                                     nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1, dilation=1))
            print(deconv, channel)
            self.sft_lv1[deconv] = motion_var_SFT(channel, channel, 1)

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel, padding=1, dilation=1)

            #if i != 3:
            #    self.adjust[deconv] = 5

    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = pac_result
                residual_feature = x
            x = self.SFT_fusion(x, motion_feature[3-i], self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            styled_feature = self.conv_up[deconv](x + residual_feature)

            if i == 3:
                pac_result = self.pac[deconv](styled_feature)

            elif i == 2:
                pac_result = self.pac[deconv](styled_feature + ef_lv2)

            else:
                pac_result = self.pac[deconv](styled_feature + ef_lv1)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result + input
        #self.ref[deconv](pac_result, input)


class FPN_Decoder_SFT_Atrous_Unet_PAC(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.adjust = nn.ModuleDict()
        #self.motion_channel = 3

        channel_list = [opt.channel * 2**(i) if i < 3 else opt.channel*4 for i in range(3)]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1, dilation=1),
                                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                        nn.LeakyReLU(0.2, True))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1, dilation=1),
                        RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                        nn.LeakyReLU(0.2, True))

            if i != 1:
                self.pac[deconv] = nn.Sequential(RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                     get_norm_layer(opt.Norm, channel, padding=1, dilation=1),
                                     nn.Conv2d(channel, channel_list[i-2], kernel_size=3, stride=1, padding=1, dilation=1),
                                     get_norm_layer(opt.Norm, channel_list[i-2], padding=1, dilation=1))
            else:
                self.pac[deconv] = nn.Sequential(RES_ASPP_Block(in_c = channel, out_c = channel, stride_len = 1),
                                     get_norm_layer(opt.Norm, channel, padding=1, dilation=1),
                                     nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1, dilation=1))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel, padding=1, dilation=1)

            #if i != 3:
            #    self.adjust[deconv] = 5

    
    def SFT_fusion(self, x, network_transform):
        return x + network_transform(x)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.feature_list = []
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x * motion_feature['patch_1']
            else:
                x = pac_result
                residual_feature = x

            x = self.SFT_fusion(x, self.non_linear_transform_lv1[deconv])
            styled_feature = self.conv_up[deconv](x + residual_feature)

            if i == 3:
                pac_result = self.pac[deconv](styled_feature)

            elif i == 2:
                pac_result = self.pac[deconv](styled_feature + ef_lv2)

            else:
                pac_result = self.pac[deconv](styled_feature + ef_lv1)
            
            self.feature_list.append(pac_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result + input
        #self.ref[deconv](pac_result, input)


class FPN_Decoder_motion_patch(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.residual_lv1 = nn.ModuleDict()
        self.residual_lv2 = nn.ModuleDict()
        self.adjust = nn.ModuleDict()
        #self.motion_channel = 3

        channel_list = [opt.channel * 2**(i) if i < 3 else opt.channel*4 for i in range(3)]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, 3, kernel_size = 3, padding=1, dilation=1),
                                        nn.LeakyReLU(0.2, True))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
                        nn.Conv2d(channel_list[i-2], channel_list[i-2], kernel_size=3,padding=1, dilation=1),
                        nn.LeakyReLU(0.2, True))

            if i != 1:
                self.residual_lv1[deconv] = nn.Sequential(get_norm_layer(opt.Norm, channel, padding=1, dilation=1))
                self.residual_lv2[deconv] = nn.Sequential(nn.Conv2d(channel, channel_list[i-2], kernel_size=3, stride=1, padding=1, dilation=1),
                                     get_norm_layer(opt.Norm, channel_list[i-2], padding=1, dilation=1))
            else:
                self.residual_lv1[deconv] = nn.Sequential(get_norm_layer(opt.Norm, channel, padding=1, dilation=1))
                self.residual_lv2[deconv] = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel, padding=1, dilation=1)


    
    def SFT_fusion(self, x, network_transform):
        return x + network_transform(x)
        
    def forward(self, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####

        styled_feature = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = motion_feature[i-1]
            else:
                x = styled_feature
                residual_feature = x

            x = self.SFT_fusion(motion_feature[3-i], self.non_linear_transform_lv1[deconv])
            res_lv1 = self.residual_lv1[deconv](x)
            result = self.residual_lv2[deconv](res_lv1 + motion_feature[3-i])
            styled_feature = self.conv_up[deconv](result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return styled_feature + input
        #self.ref[deconv](pac_result, input)



class FPN_Decoder_DCN(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.motion_channel = opt.channel * 4 #* (opt.Recurrent_times - 1)
        

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = opt.channel
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = opt.channel*2*(i-1)
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))
                
                self.sft_pac[deconv] = motion_localize_DCN(channel//2, self.motion_channel, 2**(3-i+1))

            if i != 1:
                self.pac[deconv] = PacConv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = motion_localize_DCN(channel, self.motion_channel, 2**(3-i))
            self.sft_lv2[deconv] = motion_localize_DCN(channel, self.motion_channel, 2**(3-i))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)

    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.feature_list = []

        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = pac_result
                residual_feature = x

            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])
            
            styled_feature = self.conv_up[deconv](x + residual_feature)
            if i == 3:
                pac_result = self.pac[deconv](styled_feature, ef_lv2)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)
                
            elif i == 2:
                pac_result = self.pac[deconv](styled_feature, ef_lv1)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)

            else:
                pac_result = self.pac[deconv](styled_feature)
            
            self.feature_list.append(pac_result)
        

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result 



    # 1-2-4
class FPN_Decoder_SFT_fusion(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()
        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = opt.channel
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = opt.channel*2*(i-1)
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))
                
                self.sft_pac[deconv] = SFT_DCN(channel//2, self.motion_channel, 2**(i - 2))

            if i != 1:
                self.pac[deconv] = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = SFT_DCN(channel, self.motion_channel, 2**(i - 1))
            self.sft_lv2[deconv] = SFT_DCN(channel, self.motion_channel, 2**(i - 1))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self,x, ef_lv2, ef_lv1, motion_feature, input):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.feature_list = []
        pac_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = pac_result
                residual_feature = x
    
            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            x = self.SFT_fusion(x, motion_feature, self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])

            styled_feature = self.conv_up[deconv](x + residual_feature)
            if i == 3:
                pac_result = self.pac[deconv](styled_feature)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)
                
            elif i == 2:
                pac_result = self.pac[deconv](styled_feature)
                pac_result = self.sft_pac[deconv](pac_result, motion_feature)

            else:
                pac_result = self.pac[deconv](styled_feature)
            
            self.feature_list.append(pac_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result