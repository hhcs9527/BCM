import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_modules.attention_zoo import *

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