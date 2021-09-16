import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models import *

class DMPHN_1_2_4_reblur(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_lv1 = Encoder_reblur(32)
        self.encoder_lv2 = Encoder()#_reblur(32)    
        self.encoder_lv3 = Encoder()#_reblur(32)
        
        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()    
        self.decoder_lv3 = Decoder()

    def forward(self, input):
        H = input.size(2)          
        W = input.size(3)
        self.reblur_filters = []
         
        images_lv1 = input
        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        ## level 3

        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        
        '''
        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1['encode_feature'], feature_lv3_2['encode_feature']), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3['encode_feature'], feature_lv3_4['encode_feature']), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)

        feature_lv3_1['reblur_filter'] = torch.cat((feature_lv3_1['reblur_filter']), dim = 1)
        feature_lv3_2['reblur_filter'] = torch.cat((feature_lv3_2['reblur_filter']), dim = 1)
        feature_lv3_3['reblur_filter'] = torch.cat((feature_lv3_3['reblur_filter']), dim = 1)
        feature_lv3_4['reblur_filter'] = torch.cat((feature_lv3_4['reblur_filter']), dim = 1)
        
        feature_lv3_reblur_filter_top = torch.cat((feature_lv3_1['reblur_filter'], feature_lv3_2['reblur_filter']), 3)
        feature_lv3_reblur_filter_bot = torch.cat((feature_lv3_3['reblur_filter'], feature_lv3_4['reblur_filter']), 3)
        feature_lv3_reblur_filter = torch.cat((feature_lv3_reblur_filter_top, feature_lv3_reblur_filter_bot), 2) 
        split = list(torch.split(feature_lv3_reblur_filter, 9, dim = 1)) 
        '''
        

        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        ## level 2
        
        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3

        residual_lv2 = self.decoder_lv2(feature_lv2)
        '''
        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1['encode_feature'], feature_lv2_2['encode_feature']), 2) + feature_lv3

        feature_lv2_1['reblur_filter'] = torch.cat((feature_lv2_1['reblur_filter']), dim = 1) 
        feature_lv2_2['reblur_filter'] = torch.cat((feature_lv2_2['reblur_filter']), dim = 1) 

        feature_lv2_reblur_filter = torch.cat((feature_lv2_1['reblur_filter'], feature_lv2_2['reblur_filter']), 2)    
        split = list(torch.split(feature_lv2_reblur_filter, 9, dim = 1))
        self.reblur_filters += split

        residual_lv2 = self.decoder_lv2(feature_lv2)
        '''

        ## level 1
        '''
        feature_lv1_1 = self.encoder_lv1(images_lv1 + residual_lv2)
        feature_lv1 = feature_lv1_1 + feature_lv2

        deblur_image = self.decoder_lv1(feature_lv1)   
        '''
        feature_lv1_1 = self.encoder_lv1(images_lv1 + residual_lv2)
        feature_lv1 = feature_lv1_1['encode_feature'] + feature_lv2

        feature_lv1_1['reblur_filter'] = torch.cat((feature_lv1_1['reblur_filter']), dim = 1)
        split = list(torch.split(feature_lv1_1['reblur_filter'], 9, dim = 1))
        self.reblur_filters += split

        deblur_image = self.decoder_lv1(feature_lv1)  
        #'''    

        return {'deblur' : deblur_image, 'reblur_filter' : self.reblur_filters}


class DMPHN_1_2_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()    
        self.encoder_lv3 = Encoder()
        
        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()    
        self.decoder_lv3 = Decoder()

    def forward(self, input):
        H = input.size(2)          
        W = input.size(3)
        self.reblur_filters = []
         
        images_lv1 = input
        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        ## level 3
        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        

        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        ## level 2
        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3

        residual_lv2 = self.decoder_lv2(feature_lv2)

        ## level 1
        feature_lv1_1 = self.encoder_lv1(images_lv1 + residual_lv2)
        feature_lv1 = feature_lv1_1 + feature_lv2

        deblur_image = self.decoder_lv1(feature_lv1)      

        return deblur_image

