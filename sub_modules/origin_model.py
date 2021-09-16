import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_modules.attention_zoo import *
from sub_modules.reblur_attention_zoo import *

class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
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
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        low = x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        mid = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        high = x
        #return x
        return {'encode_feature' : x}#, 'low' : low, 'mid' : mid, 'high' : high}

class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self,x):  
        x = x['encode_feature']      
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x


class Decoder_motion_DCN(nn.Module):
    '''
        motion downsample and slice them
    '''
    def __init__(self, opt): 
        super().__init__()   
        self.motion_channel = 3*3*3

        self.DCN_layer_13 = motion_Dynamic_filter(opt.channel * 4, self.motion_channel, 4)
        self.DCN_layer_14 = motion_Dynamic_filter(opt.channel * 4, self.motion_channel, 4)
        self.DCN_layer_17 = motion_Dynamic_filter(opt.channel * 2, self.motion_channel, 2)
        self.DCN_layer_18 = motion_Dynamic_filter(opt.channel * 2, self.motion_channel, 2)
        self.DCN_layer_21 = motion_Dynamic_filter(opt.channel * 1, self.motion_channel, 1)
        self.DCN_layer_22 = motion_Dynamic_filter(opt.channel * 1, self.motion_channel, 1)

        # Deconv3
        self.layer13 = nn.Sequential(
            #nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            #nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            #nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            #nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            #nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            #nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)


    def forward(self,x, motion):       
        #Deconv3
        #x = self.layer13(x) + x
        #x = self.layer14(x) + x
        x = self.layer13(self.DCN_layer_13(x, motion)) + x
        x = self.layer14(self.DCN_layer_14(x, motion)) + x
        x = self.layer16(x)                
        #Deconv2
        #x = self.layer17(x) + x
        #x = self.layer18(x) + x
        x = self.layer17(self.DCN_layer_17(x, motion)) + x
        x = self.layer18(self.DCN_layer_18(x, motion)) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(self.DCN_layer_21(x, motion)) + x
        #x = self.layer21(x) + x
        #x = self.layer22(x) + x
        x = self.layer22(self.DCN_layer_22(x, motion)) + x
        x = self.layer24(x)
        return x



class Encoder_reblur(nn.Module):
    def __init__(self, opt):
        super().__init__()
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
        #self.SP_reblur_attention_conv1 = SP_reblur_attention(opt, opt.channel, 1)

        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #self.SP_reblur_attention_conv2 = SP_reblur_attention(opt, opt.channel * 2, 2)

        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.SP_reblur_attention_conv3 = BS_attention(opt, opt.channel * 4, 4)

    def forward(self, x):
        blur = x
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #x = self.SP_reblur_attention_conv1(x)
        #self.reblur_filter.append(x['reblur_filter'])
        #x = x['attention_feature']

        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #x = self.SP_reblur_attention_conv2(x)
        #self.reblur_filter.append(x['reblur_filter'])
        #x = x['attention_feature']

        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        x = self.SP_reblur_attention_conv3(x)
        self.reblur_filter.append(x['reblur_filter'])
        x = x['attention_feature']

        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter} 


class space_to_depth(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = nn.functional.unfold(x, self.ratio, stride = self.ratio)
        return unfolded_x.view(n, c * self.ratio ** 2, h // self.ratio, w // self.ratio)


class Encoder_double_reblur(nn.Module):
    def __init__(self, opt):
        super().__init__()
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
        #self.SP_reblur_attention_conv1 = BS_attention(opt, opt.channel, 1)

        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #self.SP_reblur_attention_conv2 = BS_attention(opt, opt.channel * 2, 2)

        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        #self.SP_reblur_attention_conv3_1 = BS_attention(opt, opt.channel * 4, 4)
        #self.layer10 = nn.Sequential(
        #    nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
        #    )
        #self.SP_reblur_attention_conv3_2 = BS_attention(opt, opt.channel * 4, 4)
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.SP_reblur_attention_conv3_3 = BS_attention(opt, opt.channel * 4, 4)

        self.k = 3
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel * 4, opt.channel, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel * 4, opt.channel * 2, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )

    def forward(self, x):
        blur = x
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #x = self.SP_reblur_attention_conv1(x)
        #self.reblur_filter.append(x['reblur_filter'])
        #x = x['attention_feature']

        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #x = self.SP_reblur_attention_conv2(x)
        #self.reblur_filter.append(x['reblur_filter'])
        #x = x['attention_feature']

        #Conv3
        x = self.layer9(x)    
        #x = self.layer10(x) + x
        x = self.layer11(x) + x 
        x = self.SP_reblur_attention_conv3_3(x)
        self.reblur_filter.append(x['reblur_filter'])
        x = x['attention_feature']

        residual_attention_featurex2 = self.residual_attention_featurex2(x)
        residual_attention_featurex4 = self.residual_attention_featurex4(x)

        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4} 


class Decoder_double_reblur_front(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + encode_dict['residual_attention_featurex2'] + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + encode_dict['residual_attention_featurex4'] + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

class Decoder_triple_reblur(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x + encode_dict['residual_attention_featurex2_1']
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x + encode_dict['residual_attention_featurex4_1']
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

class Decoder_triple_reblur_front(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        #Deconv3
        x = x + encode_dict['residual_attention_featurex4']
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = x + encode_dict['residual_attention_featurex2']
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = x + encode_dict['residual_attention_featurex1']
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

class RFDB_Encoder_reblur_end(nn.Module):
    def __init__(self, opt, compression_ratio = 2):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.input_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.RFD_light_lv1 = RFD_block(opt.channel, compression_ratio = self.compression_ratio)
        self.down_sample_lv1 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.RFD_light_lv2 = RFD_block(opt.channel*2, compression_ratio = self.compression_ratio)
        self.down_sample_lv2= nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.RFD_light_lv3 = RFD_block(opt.channel*4, compression_ratio = self.compression_ratio)
        self.SP_reblur_attention = BS_attention(opt, opt.channel * 4, 4)

        self.run = nn.Sequential(
                    self.input_layer,
                    self.RFD_light_lv1,
                    self.down_sample_lv1,
                    self.RFD_light_lv2,
                    self.down_sample_lv2,
                    self.RFD_light_lv3,
                    self.SP_reblur_attention,
                    )
        self.k = 3
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel * 4, opt.channel, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel * 4, opt.channel * 2, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )

    def forward(self, input):
        self.reblur_filter = []
        x = self.run(input)
        self.reblur_filter.append(x['reblur_filter'])
        x = x['attention_feature']

        residual_attention_featurex2 = self.residual_attention_featurex2(x)
        residual_attention_featurex4 = self.residual_attention_featurex4(x)

        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4} 


class RFDB_Encoder_reblur_front(nn.Module):
    def __init__(self, opt, compression_ratio = 2):
        super().__init__()
        # layers
        self.compression_ratio = compression_ratio
        self.input_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.RFD_light_lv1 = RFD_block(opt.channel, compression_ratio = self.compression_ratio)
        self.down_sample_lv1 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.RFD_light_lv2 = RFD_block(opt.channel*2, compression_ratio = self.compression_ratio)
        self.down_sample_lv2= nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.RFD_light_lv3 = RFD_block(opt.channel*4, compression_ratio = self.compression_ratio)

        # attentions
        self.SP_reblur_attention = BS_attention(opt, opt.channel, 1)

        self.run_head = nn.Sequential(
                    self.input_layer,
                    self.RFD_light_lv1,
                    self.SP_reblur_attention,
                    )

        self.run_tail = nn.Sequential(
                    self.down_sample_lv1,
                    self.RFD_light_lv2,
                    self.down_sample_lv2,
                    self.RFD_light_lv3,
                    )

        self.k = 1
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )


    def forward(self, input):
        self.reblur_filter = []
        x = self.run_head(input)
        self.reblur_filter.append(x['reblur_filter'])
        attention_feature = x['attention_feature']
        x = self.run_tail(attention_feature)

        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)
        
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter, 'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4} 


class RFDB_Encoder_reblur(nn.Module):
    def __init__(self, opt, compression_ratio = 2):
        super().__init__()
        # layers
        self.compression_ratio = compression_ratio
        self.input_layer = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.RFD_light_lv1 = RFD_block(opt.channel, compression_ratio = self.compression_ratio)
        self.down_sample_lv1 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.RFD_light_lv2 = RFD_block(opt.channel*2, compression_ratio = self.compression_ratio)
        self.down_sample_lv2= nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.RFD_light_lv3 = RFD_block(opt.channel*4, compression_ratio = self.compression_ratio)

        # attentions
        self.reblur_local_head = BS_attention(opt, opt.channel, 1)
        self.reblur_local_mid = global_attention(opt, opt.channel*2, 2)#BS_attention(opt, opt.channel*2, 2)
        self.reblur_global = global_attention(opt, opt.channel*4, 4)

        self.run_head = nn.Sequential(
                    self.input_layer,
                    self.RFD_light_lv1,
                    self.reblur_local_head,
                    )
        
        self.run_mid = nn.Sequential(
                    self.down_sample_lv1,
                    self.RFD_light_lv2,
                    self.reblur_local_mid,
                    )

        self.run_tail = nn.Sequential(
                    self.down_sample_lv2,
                    self.RFD_light_lv3,
                    self.reblur_global,
                    )

        self.k = 1
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )

        self.residual_attention_featurex4_1 = nn.Sequential(
                                        nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel * 4, opt.channel, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2_1 = nn.Sequential(
                                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel * 4, opt.channel * 2, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )

    def forward(self, input):
        self.reblur_filter = []
        # head
        x = self.run_head(input)
        self.reblur_filter.append(x['reblur_filter'])
        x = x['attention_feature']

        # mid
        x = self.run_mid(x)#attention_feature)
        #self.reblur_filter.append(x['reblur_filter'])
        x = x['attention_feature']

        # tail
        x = self.run_tail(x)
        #self.reblur_filter.append(x['reblur_filter'])
        x = x['attention_feature']

        #residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        #residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        #residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)


        residual_attention_featurex2_1 = self.residual_attention_featurex2_1(x)
        residual_attention_featurex4_1 = self.residual_attention_featurex4_1(x)

        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter, 
                #'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
                'residual_attention_featurex2_1' : residual_attention_featurex2_1, 'residual_attention_featurex4_1' : residual_attention_featurex4_1} 


class RFDB_Decoder_reblur(nn.Module):
    def __init__(self, opt, compression_ratio = 2):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.RFD_light_lv3 = nn.Sequential(
                            RFD_block(opt.channel*4, compression_ratio = self.compression_ratio),
                            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                            nn.Conv2d(opt.channel * 4, opt.channel * 2, kernel_size=3, stride=1, padding=1),
                            #nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1),
                            )
        self.RFD_light_lv2 = nn.Sequential(
                            RFD_block(opt.channel*2, compression_ratio = self.compression_ratio),
                            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                            nn.Conv2d(opt.channel * 2, opt.channel * 1, kernel_size=3, stride=1, padding=1),
                            #nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1),
                            )
        self.RFD_light_lv1 = nn.Sequential(
                            RFD_block(opt.channel, compression_ratio = self.compression_ratio),
                            nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1),
                            )

    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        # Deconv 3
        x = self.RFD_light_lv3(x + encode_dict['residual_attention_featurex4'])
        # Deconv 2
        x = self.RFD_light_lv2(x + encode_dict['residual_attention_featurex2'])
        # Deconv 1
        x = self.RFD_light_lv1(x + encode_dict['residual_attention_featurex1'])
        return x

class RFDB_Decoder_reblur_part(nn.Module):
    def __init__(self, opt, compression_ratio = 2):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.RFD_light_lv3 = nn.Sequential(
                            RFD_block(opt.channel*4, compression_ratio = self.compression_ratio),
                            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                            nn.Conv2d(opt.channel * 4, opt.channel * 2, kernel_size=3, stride=1, padding=1),
                            #nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1),
                            )
        self.RFD_light_lv2 = nn.Sequential(
                            RFD_block(opt.channel*2, compression_ratio = self.compression_ratio),
                            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                            nn.Conv2d(opt.channel * 2, opt.channel * 1, kernel_size=3, stride=1, padding=1),
                            #nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1),
                            )
        self.RFD_light_lv1 = nn.Sequential(
                            RFD_block(opt.channel, compression_ratio = self.compression_ratio),
                            nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1),
                            )

    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        # Deconv 3
        x = self.RFD_light_lv3(x) + encode_dict['residual_attention_featurex2']
        # Deconv 2
        x = self.RFD_light_lv2(x) + encode_dict['residual_attention_featurex4']
        # Deconv 1
        x = self.RFD_light_lv1(x)
        return x

class BR_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer = RFD_block_reblur(opt, opt.channel)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )

    def forward(self, x):
        self.feature_maps = {}
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        #Conv2
        x = self.layer5(attention_feature)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        k = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen}
        #f'lv{self.level}_low': self.reblur_filter[0]}#attention_feature}

class BRR_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer = RFD_block_reblur(opt, opt.channel)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.k = 3
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel*4, opt.channel * 4, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel*2, opt.channel * 2, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding = 1),
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )        
    def forward(self, x):
        self.feature_maps = {}
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        #Conv2
        x = self.layer5(attention_feature)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        k = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        c = x
        #return x
        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen}
        #f'lv{self.level}_low': self.reblur_filter[0]}#attention_feature}


class BRRDCN_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer = RFD_block_reblurDCN(opt, opt.channel)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.k = 3
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel*4, opt.channel * 4, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel*2, opt.channel * 2, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding = 1),
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )        
    def forward(self, x):
        self.feature_maps = {}
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        #Conv2
        x = self.layer5(attention_feature)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        k = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        c = x
        #return x
        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen}
        #f'lv{self.level}_low': self.reblur_filter[0]}#attention_feature}


class BER_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer = RFD_block_reblur(opt, opt.channel)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.k = 3
        self.residual_attention_featurex4f = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel*4, opt.channel * 4, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2f = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel*2, opt.channel * 2, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel*4, opt.channel * 4, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel*2, opt.channel * 2, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding = 1),
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )        
    def forward(self, x):
        self.feature_maps = {}
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        residual_attention_featurex2f = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4f = self.residual_attention_featurex4(attention_feature)

        #Conv2
        x = self.layer5(attention_feature)
        x = x + residual_attention_featurex2f
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        k = x
        #Conv3
        x = self.layer9(x)
        x = x + residual_attention_featurex4f    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        c = x
        #return x
        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)

        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen}
        #f'lv{self.level}_low': self.reblur_filter[0]}#attention_feature}


class RFDB_Encoder_multi_scale(nn.Module):
    def __init__(self, opt, level = 1):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer1 = RFD_block_reblur_multi_scale(opt, opt.channel, ratio = 1)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        ###
        self.RFDB_layer2 = RFD_block_reblur_multi_scale(opt, opt.channel*2, ratio = 2)
        ###
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        ###
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        ###
    def forward(self, x):
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer1(x)
        self.reblur_filter += x['reblur_filter']
        x = x['encode_feature']
        residual_1 = x
        #Conv2
        x = self.layer5(x)
        x = self.RFDB_layer2(x)
        self.reblur_filter += x['reblur_filter']
        x = x['encode_feature']
        residual_2 = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        residual_4 = x
        #return x
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_1, 'residual_attention_featurex2' : residual_2, 'residual_attention_featurex4' : residual_4,
        f'lv{self.level}_low': residual_1, f'lv{self.level}_mid': residual_2, f'lv{self.level}_high': residual_4}


class RFDB_Encoder(nn.Module):
    def __init__(self, opt, level = 1):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer = RFD_block_reblur_old(opt, opt.channel)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.k = 1
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = self.k, padding = self.k//2),
                                        nn.LeakyReLU(0.2),
                                        )        
    def forward(self, x):
        self.feature_maps = {}
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        #Conv2
        x = self.layer5(attention_feature)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        #return x
        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen}
        #f'lv{self.level}_low': self.reblur_filter[0]}
        #f'lv{self.level}_low': attention_feature}

class BRR_Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        #Deconv3
        x = x + encode_dict['residual_attention_featurex4']
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = x + encode_dict['residual_attention_featurex2']
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = x + encode_dict['residual_attention_featurex1']
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x


class BRR_Decoder_back(nn.Module):
    def __init__(self, opt, level = 1):
        self.level = level
        super().__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(opt.channel * 4, opt.channel * 2, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(opt.channel * 2, opt.channel, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.RFDB_layer = RFD_block_reblur(opt, opt.channel)
        self.layer24 = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)
        
    def forward(self, encode_dict):
        x = encode_dict['encode_feature']
        self.reblur_filter = []
        #Deconv3
        x = x + encode_dict['residual_attention_featurex4']
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = x + encode_dict['residual_attention_featurex2']
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = x + encode_dict['residual_attention_featurex1']
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        x = self.layer24(attention_feature)
        return {'decode_feature' : x, 'reblur_filter' : self.reblur_filter,
        f'lv{self.level+1}_low': conv, f'lv{self.level+1}_mid': blur, f'lv{self.level+1}_high': sharpen}

class BRRM_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.RFDB_layer = RFD_block_reblur_mask(opt, opt.channel)
        ###
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.k = 3
        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel*4, opt.channel * 4, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel*2, opt.channel * 2, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding = 1),
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )        
    def forward(self, x):
        self.feature_maps = {}
        self.reblur_filter = []
        #Conv1
        x = self.layer1(x)
        x = self.RFDB_layer(x)
        self.reblur_filter += x['reblur_filter']
        attention_feature = x['encode_feature']
        sharpen = x['sharpen']
        blur = x['blur']
        conv = x['conv']
        mask = x['blurmask']
        #Conv2
        x = self.layer5(attention_feature)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        k = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        c = x
        #return x
        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)
        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen, 'blurmask': mask}
class REncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
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
        #Conv2
        self.layer5 = nn.Conv2d(opt.channel, opt.channel * 2, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 2, opt.channel * 2, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(opt.channel * 2, opt.channel * 4, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(opt.channel * 4, opt.channel * 4, kernel_size=3, padding=1)
            )

        self.residual_attention_featurex4 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 4, kernel_size = 3, padding = 1, stride = 4),
                                        nn.Conv2d(opt.channel*4, opt.channel * 4, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex2 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel * 2, kernel_size = 3, padding = 1, stride = 2),
                                        nn.Conv2d(opt.channel*2, opt.channel * 2, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )
        self.residual_attention_featurex1 = nn.Sequential(
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 3, padding = 1),
                                        nn.Conv2d(opt.channel, opt.channel, kernel_size = 1, padding = 0, stride = 1),
                                        nn.LeakyReLU(0.2),
                                        )    

    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        low = x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        mid = x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        high = x

        residual_attention_featurex1 = self.residual_attention_featurex1(low)
        residual_attention_featurex2 = self.residual_attention_featurex2(low)
        residual_attention_featurex4 = self.residual_attention_featurex4(low)

        #return x
        return {'encode_feature' : x, 'low' : low, 'mid' : mid, 'high' : high,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4}
