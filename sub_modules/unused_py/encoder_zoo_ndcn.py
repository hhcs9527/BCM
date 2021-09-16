import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sub_modules.attention_zoo import *
from sub_modules.component_block import *
from sub_modules.bifpn_block import *


class PAN_encoder(nn.Module):
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

        self.conv_up = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_predict = nn.ModuleDict()
        self.FPN_level = 4
        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]

        for i in range(1,self.FPN_level):
            level = f'level_{i}'  
            self.conv_lateral[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=1, stride=1)
            self.conv_predict[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, stride=1, padding=1)

            if i != self.FPN_level-1:
                self.conv_up[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i], kernel_size=3, stride=2, padding=1)


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

        # PA aggregation
        FPN_info = {'fpn_feature_lv1' : ef_lv1, 'fpn_feature_lv2' : ef_lv2, 'fpn_feature_lv3' : ef_lv3,}
        PAN_info = {}
        for i in range(1,self.FPN_level):
            level = f'level_{i}'
            bottom_level = f'level_{i-1}'
            fpn_feature = f'fpn_feature_lv{i}'
            bottom_fpn_feature = f'fpn_feature_lv{i-1}'
            from_lateral = self.conv_lateral[level](FPN_info[fpn_feature])
            if i != 1:
                from_bottom = self.conv_up[bottom_level](FPN_info[bottom_fpn_feature])
                PAN_info[level] = self.conv_predict[level](from_lateral + from_bottom)
            else:
                PAN_info[level] = self.conv_predict[level](from_lateral)


        return PAN_info[level]


