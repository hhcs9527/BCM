import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sub_modules.attention_zoo import *
from sub_modules.component_block import *
from sub_modules.bifpn_block import *
from DCNv2.DCN.dcn_v2 import DCN


class SP_FPN_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.SP = nn.ModuleDict()
        self.DCN_extract = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.Run_sequence = nn.ModuleDict()
        self.FPN_level = 6
        self.num_bifpn = 0
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.num_sp_attention = 2
        self.motion_channel = 3 * (opt.per_pix_kernel **2) * opt.Recurrent_times

        out_channel_list = [opt.channel * 2 ** (i//2) for i in range(self.FPN_level)]
        in_channel_list = [3] + [opt.channel * 2 ** (i//2) for i in range(self.FPN_level-1)]
        stride_list = [2, 1, 2, 1, 1, 1]
        kernel = 3

        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            in_channel = in_channel_list[i-1]
            out_channel = out_channel_list[i-1]
            stride_length = stride_list[i-1]

            #if i in [1, 2]:
            #    att_mod_list = nn.ModuleDict()
            #    for j in range(self.num_sp_attention):
            #        att = f'att_{j+1}'
            #        att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, 2)
        
            #    self.sp_attention[level] = att_mod_list

            #elif i in [3, 4, 5, 6]:
            #    att_mod_list = nn.ModuleDict()
            #    for j in range(self.num_sp_attention):
            #        att = f'att_{j+1}'
            #        att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, 4)
        
            #    self.sp_attention[level] = att_mod_list


            self.SP[level] = StripPooling(opt, in_channel, strip_size = 1)
            self.DCN_extract[level] = Residual_module(nn.Sequential(
                Residual_module(
                DCN(in_channel, in_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2)),
                Base_Res_Block(opt, in_channel),
                Residual_module(
                DCN(in_channel, in_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2)),))

            self.conv_up[level] = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding=1, stride = stride_length)
            self.Residual[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel), Base_Res_Block(opt, out_channel)))
            self.Run_sequence[level] = nn.Sequential(
                self.SP[level],
                self.DCN_extract[level],
                self.conv_up[level], self.Residual[level],
                DCN(out_channel, out_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),)
    
        
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            #if i % 2 == 1:
            #    self.bifpn[level] = BiFPN_attention_layer(opt, out_channel_list, stride_list)
            #else:
            self.bifpn[level] = BiFPN_layer(opt, out_channel_list, stride_list)



    def forward(self, x, motion):
        # FPN goes up from Conv 1 - 4
        # Goes up
        FPN_info = {}
        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'
            if i == 1:
                FPN_info[fpn_feature] = self.Run_sequence[level](x)
            else:
                FPN_info[fpn_feature] = self.Run_sequence[level](FPN_info[last_fpn_feature])

        #for i in range(1, self.FPN_level + 1):
        #    level = f'level_{i}'
        #    fpn_feature = f'fpn_feature_lv{i}'
        #    for j in range(self.num_sp_attention):
        #        att = f'att_{j+1}'
        #        FPN_info[fpn_feature] = self.sp_attention[level][att](FPN_info[fpn_feature], motion)

        pred_bifpn = {}
        for i in range(2,self.FPN_level+1):
            fpn_feature = f'fpn_feature_lv{i}'
            pred_bifpn[fpn_feature] = FPN_info[fpn_feature]    

        # Goes down (BiFPN part)
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            if i == 1:
                pred_bifpn = self.bifpn[level](FPN_info, motion)
            else:
                pred_bifpn = self.bifpn[level](pred_bifpn, motion)
        
        #return FPN_info, pred_bifpn

        return FPN_info, pred_bifpn


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
        #ef_lv2 = self.pred_ef_lv2(self.lateral_conv2(ef_lv2) + self.up_to_conv2(ef_lv3))
        #ef_lv1 = self.pred_ef_lv1(self.lateral_conv1(ef_lv1) + self.up_to_conv1(ef_lv2))

        #return ef_lv1, ef_lv2, ef_lv3
        return ef_lv3


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
                PAN_info[fpn_feature] = self.conv_predict[level](from_lateral + from_bottom)
            else:
                PAN_info[fpn_feature] = self.conv_predict[level](from_lateral)
        

        return {'encode_feature' : PAN_info}


class PAN_group_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        self.layer2 = get_norm_layer(opt.Norm, opt.channel)
        self.layer3 = get_norm_layer(opt.Norm, opt.channel)
        #self.layer2 = get_DCN_norm_layer(opt.Norm, opt.channel)
        #self.layer3 = get_DCN_norm_layer(opt.Norm, opt.channel)        
        # Lateral to Conv1
        self.lateral_conv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=1)
        # ef_lv1
        self.pred_ef_lv1 = nn.Conv2d(opt.channel, opt.channel, kernel_size=3, padding=1)

        # Conv2 
        self.layer5 = nn.Conv2d(opt.channel, opt.channel*2, kernel_size=3, stride=2, padding=1)
        self.layer6 = get_norm_layer(opt.Norm, opt.channel*2)
        self.layer7 = get_norm_layer(opt.Norm, opt.channel*2)
        #self.layer6 = get_DCN_norm_layer(opt.Norm, opt.channel*2)
        #self.layer7 = get_DCN_norm_layer(opt.Norm, opt.channel*2)        
        # Lateral to Conv2
        self.lateral_conv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel*2, opt.channel, kernel_size=3, padding=1)
                                        )
        #nn.ConvTranspose2d(opt.channel*2, opt.channel, kernel_size=3, stride=2, padding=1, output_padding = 1)
        # ef_lv2
        self.pred_ef_lv2 = nn.Conv2d(opt.channel*2, opt.channel*2, kernel_size=3, padding=1)

        # Conv3
        self.layer9 = nn.Conv2d(opt.channel*2, opt.channel*4, kernel_size=3, stride=2, padding=1)
        self.layer10 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer11 = get_norm_layer(opt.Norm, opt.channel*4)
        #self.layer10 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        #self.layer11 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv3
        self.lateral_conv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv2
        self.up_to_conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(opt.channel*4, opt.channel*2, kernel_size=3, padding=1)
                                        )
        #nn.ConvTranspose2d(opt.channel*4, opt.channel*2, kernel_size=3, stride=2, padding=1, output_padding = 1)
        # ef_lv3
        self.pred_ef_lv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, padding=1)

        # Conv4
        self.layer12 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=1, padding=1)
        self.layer13 = get_norm_layer(opt.Norm, opt.channel*4)
        self.layer14 = get_norm_layer(opt.Norm, opt.channel*4)
        #self.layer13 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        #self.layer14 = get_DCN_norm_layer(opt.Norm, opt.channel*4)
        # Lateral to Conv4
        self.lateral_conv4 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=1)
        # convtrans to Conv3
        self.up_to_conv3 = nn.Conv2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=1, padding=1)
        #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #nn.ConvTranspose2d(opt.channel*4, opt.channel*4, kernel_size=3, stride=2, padding=1, output_padding = 1)

        #self.predict_deblur = nn.Conv2d(opt.channel, 3, kernel_size=3, padding=1)

        self.conv_up = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_predict = nn.ModuleDict()
        self.attention_conv_up = nn.ModuleDict()
        self.FPN_level = 4
        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [1, 2, 4, 4]

        for i in range(1,self.FPN_level):
            level = f'level_{i}'  
            ratio = ratio_list[i-1]
            self.conv_lateral[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=1, stride=1)
            self.conv_predict[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, stride=1, padding=1)

            if i != self.FPN_level-1:
                #self.attention_conv_up[level] = SP_residual_attention(opt, out_channel_list[i-1], ratio)
                self.conv_up[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i], kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        blur = x
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

        #deblur = self.predict_deblur(ef_lv1) + blur

        # PA aggregation
        FPN_info = {'fpn_feature_lv1' : ef_lv1, 'fpn_feature_lv2' : ef_lv2, 'fpn_feature_lv3' : ef_lv3,}
        PAN_info = {}
        for i in range(1,self.FPN_level):
            level = f'level_{i}'
            bottom_level = f'level_{i-1}'
            bottom_level = f'level_{i-1}'
            fpn_feature = f'fpn_feature_lv{i}'
            bottom_fpn_feature = f'fpn_feature_lv{i-1}'
            from_lateral = self.conv_lateral[level](FPN_info[fpn_feature])
            if i != 1:
                from_bottom = self.conv_up[bottom_level](FPN_info[bottom_fpn_feature])
                PAN_info[level] = self.conv_predict[level](from_lateral + from_bottom)
            else:
                PAN_info[level] = self.conv_predict[level](from_lateral)

        PAN_dict = {'encode_feature' : PAN_info[level]} #{'deblur' : deblur}

        return PAN_dict


class BiFPN_Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.SP = nn.ModuleDict()
        self.DCN_extract = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_predict = nn.ModuleDict()
        self.Run_sequence = nn.ModuleDict()
        self.FPN_level = 4
        self.num_bifpn = 1
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.conv_from_top = nn.ModuleDict()
        self.num_sp_attention = 1
        self.motion_channel = 3 * (opt.per_pix_kernel **2) * opt.Recurrent_times
        kernel = 3

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        in_channel_list = [3, opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [1 ,2 ,4 ,4]
        stride_list = [1, 2, 2, 1] # 256, 128, 64, 64


        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            in_channel = in_channel_list[i-1]
            out_channel = out_channel_list[i-1]
            stride_length = stride_list[i-1]
            ratio = ratio_list[i-1]

            self.bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # SP_attention
            att_mod_list = nn.ModuleDict()
            if i != 1:
                for j in range(self.num_sp_attention):
                    att = f'att_{j+1}'
                    att_mod_list[att] = SP_motion_attention(opt, out_channel, self.motion_channel, ratio)

            self.sp_attention[level] = att_mod_list

            self.SP[level] = StripPooling(opt, in_channel, strip_size = 1)

            self.DCN_extract[level] = nn.Sequential(
                        DCN(in_channel, in_channel, kernel_size=(5, 5), stride=1, padding = 5//2),
                        #Res_Block('leaky', in_channel)
                        )

            self.conv_up[level] = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1, stride = stride_length)

            self.Residual[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel), Base_Res_Block(opt, out_channel)))

            self.Run_sequence[level] = nn.Sequential(
                #self.SP[level],
                self.DCN_extract[level],
                self.conv_up[level], self.Residual[level],
                nn.Conv2d(out_channel, out_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),)
         
            if i != 1:
                if i != self.FPN_level:
                    self.conv_predict[level] = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1, stride = 1)

                    if ratio_list[i-1] != 4:
                        self.conv_from_top[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                    nn.Conv2d(out_channel_list[i], out_channel, kernel_size = 3, padding = 1, stride = 1),
                                                    )
                    else:
                        self.conv_from_top[level] = nn.Sequential(
                                                    nn.Conv2d(out_channel_list[i], out_channel, kernel_size = 3, padding = 1, stride = 1),
                                                    )     

                self.conv_lateral[level] = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1, stride = 1)

        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            self.bifpn[level] = BiFPN(opt, out_channel_list, stride_list)

    def forward(self, x, motion):
        # FPN goes up from Conv 1 - 4
        # Goes up
        FPN_info = {}
        ratio_list = [1 ,2 ,4 ,4]
        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'
            if i == 1:
                FPN_info[fpn_feature] = self.Run_sequence[level](x)
            else:
                FPN_info[fpn_feature] = self.Run_sequence[level](FPN_info[last_fpn_feature])
        
        # Goes down
        FPN_down_info = {}
        for i in range(self.FPN_level, 1, -1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            top_fpn_feature = f'fpn_feature_lv{i+1}'
            if i == self.FPN_level:
                FPN_down_info[fpn_feature] = self.conv_lateral[level](FPN_info[fpn_feature])
            else:
                FPN_down_info[fpn_feature] = self.conv_predict[level](self.conv_lateral[level](FPN_info[fpn_feature]) + self.conv_from_top[level](FPN_down_info[top_fpn_feature]))

        
        if self.num_sp_attention:
            for i in range(2, self.FPN_level + 1):
                level = f'level_{i}'
                fpn_feature = f'fpn_feature_lv{i}'
                for j in range(self.num_sp_attention):
                    att = f'att_{j+1}'
                    FPN_down_info[fpn_feature] = self.sp_attention[level][att](FPN_down_info[fpn_feature], motion)
                    

        pred_bifpn = {}
        for i in range(2,self.FPN_level+1):
            fpn_feature = f'fpn_feature_lv{i}'
            pred_bifpn[fpn_feature] = FPN_down_info[fpn_feature]    

        # Goes down (BiFPN part)
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            if i == 1:
                pred_bifpn = self.bifpn[level](FPN_down_info, motion)
            else:
                pred_bifpn = self.bifpn[level](pred_bifpn, motion)
        
        return FPN_info, pred_bifpn


class BiFPN_Encoder_block(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.SP = nn.ModuleDict()
        self.DCN_extract = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_predict = nn.ModuleDict()
        self.Run_sequence = nn.ModuleDict()
        self.FPN_level = 4
        self.num_bifpn = 0
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.conv_from_top = nn.ModuleDict()
        self.num_sp_attention = 0
        self.motion_channel = 3 * (opt.per_pix_kernel **2) * opt.Recurrent_times
        self.PA_conv_up = nn.ModuleDict()
        self.PA_conv_lateral = nn.ModuleDict()
        self.PA_conv_predict = nn.ModuleDict()
        kernel = 3

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        in_channel_list = [3, opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [1 ,2 ,4 ,4]
        stride_list = [1, 2, 2, 1] # 256, 128, 64, 64


        for i in range(1, self.FPN_level+1):
            level = f'level_{i}'
            in_channel = in_channel_list[i-1]
            out_channel = out_channel_list[i-1]
            stride_length = stride_list[i-1]
            ratio = ratio_list[i-1]

            self.bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            #self.sp_attention[level] = att_mod_list

            self.SP[level] = StripPooling(opt, in_channel, strip_size = 1)

            self.DCN_extract[level] = nn.Sequential(
                        DCN(in_channel, in_channel, kernel_size=(5, 5), stride=1, padding = 5//2),
                        #Res_Block('leaky', in_channel)
                        )

            self.conv_up[level] = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1, stride = stride_length)

            self.Residual[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel), Base_Res_Block(opt, out_channel)))

            self.Run_sequence[level] = nn.Sequential(
                #self.SP[level],
                self.DCN_extract[level],
                self.conv_up[level], self.Residual[level],
                DCN(out_channel, out_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),)
         
            if i != self.FPN_level:
                self.conv_predict[level] = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1, stride = 1)

                if ratio_list[i-1] != 4:
                    self.conv_from_top[level] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                nn.Conv2d(out_channel_list[i], out_channel, kernel_size = 3, padding = 1, stride = 1),
                                                )
                else:
                    self.conv_from_top[level] = nn.Sequential(
                                                nn.Conv2d(out_channel_list[i], out_channel, kernel_size = 3, padding = 1, stride = 1),
                                                )     

            self.conv_lateral[level] = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1, stride = 1)


            if i <= self.FPN_level - 1:
                self.PA_conv_lateral[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, stride=1, padding=1)
                self.PA_conv_predict[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, stride=1, padding=1)

            if i < self.FPN_level - 1:
                self.PA_conv_up[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i], kernel_size=3, stride=2, padding=1)


    def forward(self, x, motion):
        # FPN goes up from Conv 1 - 4
        # Goes up
        FPN_info = {}
        ratio_list = [1 ,2 ,4 ,4]
        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'
            if i == 1:
                FPN_info[fpn_feature] = self.Run_sequence[level](x)
            else:
                FPN_info[fpn_feature] = self.Run_sequence[level](FPN_info[last_fpn_feature])
        
        # Goes down
        FPN_down_info = {}
        for i in range(self.FPN_level, 0, -1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            top_fpn_feature = f'fpn_feature_lv{i+1}'
            if i == self.FPN_level:
                FPN_down_info[fpn_feature] = self.conv_lateral[level](FPN_info[fpn_feature])
            else:
                FPN_down_info[fpn_feature] = self.conv_predict[level](self.conv_lateral[level](FPN_info[fpn_feature]) + self.conv_from_top[level](FPN_down_info[top_fpn_feature]))
        '''
        # PA aggregation
        PAN_info = {}
        for i in range(1, self.FPN_level):
            level = f'level_{i}'
            bottom_level = f'level_{i-1}'
            fpn_feature = f'fpn_feature_lv{i}'
            bottom_fpn_feature = f'fpn_feature_lv{i-1}'
            from_lateral = self.conv_lateral[level](FPN_down_info[fpn_feature])
            if i != 1:
                from_bottom = self.conv_up[bottom_level](FPN_down_info[bottom_fpn_feature])
                PAN_info[level] = self.conv_predict[level](from_lateral + from_bottom)
            else:
                PAN_info[level] = self.conv_predict[level](from_lateral)  
        '''

        return FPN_down_info[fpn_feature] #PAN_info[level]


# multi-scale FPN & fuse with bifpn
class multi_scale_BiFPN_Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.SP = nn.ModuleDict()
        self.DCN_extract = nn.ModuleDict()
        self.scale_blur = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_predict = nn.ModuleDict()
        self.Run_sequence = nn.ModuleDict()
        self.num_scale = 3
        self.num_bifpn = 3
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.conv_from_top = nn.ModuleDict()
        self.BiFPN_block = nn.ModuleDict()
        self.num_sp_attention = 0
        self.motion_channel = 3 * (opt.per_pix_kernel **2) * opt.Recurrent_times
        kernel = 3

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        in_channel_list = [3, opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [1 ,2 ,4 ,4]
        stride_list = [1, 2, 2, 1] # 256, 128, 64, 64
        

        for i in range(1,self.num_scale + 1):
            level = f'level_{i}' 
            self.BiFPN_block[level] = BiFPN_Encoder_block(opt)

        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            self.bifpn[level] = bifpn_multi_scale_fusion(opt, out_channel_list, stride_list)


    def forward(self, x, motion):
        # FPN goes up from Conv 1 - 4
        # Get multi-scale info from 256, 128, .....
        multi_scale_FPN_info = {}
        ratio_list = [1 ,2 ,4 ,4]
        scale_list = [2 ** i for i in range(self.num_scale)]

        for i in range(1, self.num_scale+1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            if scale_list[i-1] == 1:
                blur_input = x
            else:
                blur_input = F.interpolate(x, scale_factor = 1/scale_list[i-1] , mode = "bilinear", recompute_scale_factor=True,align_corners=True)
            multi_scale_FPN_info[fpn_feature] = self.BiFPN_block[level](blur_input, motion)
        

        pred_bifpn = {}
        for i in range(1, self.num_scale + 1):
            fpn_feature = f'fpn_feature_lv{i}'
            pred_bifpn[fpn_feature] = multi_scale_FPN_info[fpn_feature]    


        # Goes down (BiFPN part)
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            if i == 1:
                pred_bifpn = self.bifpn[level](multi_scale_FPN_info, motion)
            else:
                pred_bifpn = self.bifpn[level](pred_bifpn, motion)

        return pred_bifpn[f'fpn_feature_lv{self.num_scale}']



class Patch_Unet_Encoder(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.reblur_attention = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel, opt.channel*2, opt.channel*4]

        ratio_list = [1, 2, 4]
        for i in range(self.patch_level, 0, -1):
            channel = channel_list[i-1]
            ratio = ratio_list[i-1]

            level = f'level{i}'
            if i == 1:
                self.Patch_FE[level] = nn.Sequential(
                                        Guide_DCN_Block(opt, in_c = 3, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                        Base_Res_Block(opt, channel),
                                        Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, ratio)
                                        )

            elif i != self.patch_level:
                previous_channel = channel_list[i-2]
                self.Patch_FE[level] = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        Base_Res_Block(opt, channel),
                                        Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, ratio)
                                        )

            else:
                previous_channel = channel_list[i-2]
                self.Patch_FE[level] = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        Base_Res_Block(opt, channel),
                                        Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, ratio)
                                        )                                               

    def get_image_level(self,x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]

    def append_reblur_filter(self, reblur_dict):
        self.reblur_filter_list.append(reblur_dict['reblur_filter'])
        return reblur_dict['attention_feature']

    def forward(self, x):
        self.get_image_level(x)
        H = x.size(2)
        W = x.size(3)
        self.feature_list = []
        self.reblur_filter_list = []
        Patch_Unet_dict = {}

        for i in range(1, self.patch_level + 1):
            feature_level = f'fpn_feature_lv{i}'
            bottom_feature_level = f'fpn_feature_lv{i-1}'
            level = f'level{i}'
            

            if i == self.patch_level:
                feature_lv1 = self.Patch_FE[level](Patch_Unet_dict[bottom_feature_level])
                Patch_Unet_dict[feature_level] = feature_lv1['attention_feature']
                
                self.reblur_filter_list.append(feature_lv1['reblur_filter'])

            elif i == 2:
                feature_lv2_1 = Patch_Unet_dict[bottom_feature_level][:,:,0:int(H/2),:]
                feature_lv2_2 = Patch_Unet_dict[bottom_feature_level][:,:,int(H/2):H,:]
                feature_lv2_1 = self.Patch_FE[level](feature_lv2_1.contiguous())
                feature_lv2_2 = self.Patch_FE[level](feature_lv2_2.contiguous())
                feature_lv2 = torch.cat((feature_lv2_1['attention_feature'], feature_lv2_2['attention_feature']), 2)
                Patch_Unet_dict[feature_level] = feature_lv2

                feature_lv2_reblur_filter = torch.cat((feature_lv2_1['reblur_filter'], feature_lv2_2['reblur_filter']), 2)
                self.reblur_filter_list.append(feature_lv2_reblur_filter)


            elif i == 1:
                feature_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
                feature_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
                feature_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
                feature_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
                feature_lv3_1 = self.Patch_FE[level](feature_lv3_1.contiguous())
                feature_lv3_2 = self.Patch_FE[level](feature_lv3_2.contiguous())
                feature_lv3_3 = self.Patch_FE[level](feature_lv3_3.contiguous())
                feature_lv3_4 = self.Patch_FE[level](feature_lv3_4.contiguous())
                feature_lv3_top = torch.cat((feature_lv3_1['attention_feature'], feature_lv3_2['attention_feature']), 3)
                feature_lv3_bot = torch.cat((feature_lv3_3['attention_feature'], feature_lv3_4['attention_feature']), 3)
                feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                Patch_Unet_dict[feature_level] = feature_lv3
 
                feature_lv3_reblur_filter_top = torch.cat((feature_lv3_1['reblur_filter'], feature_lv3_2['reblur_filter']), 3)
                feature_lv3_reblur_filter_bot = torch.cat((feature_lv3_3['reblur_filter'], feature_lv3_4['reblur_filter']), 3)
                feature_lv3_reblur_filter = torch.cat((feature_lv3_reblur_filter_top, feature_lv3_reblur_filter_bot), 2)
                self.reblur_filter_list.append(feature_lv3_reblur_filter)
        
        # out size : [batch, opt.channel * 4, H, W]
        encoder_dict = {'encode_feature' : Patch_Unet_dict, 'reblur_filter' : self.reblur_filter_list} #{'deblur' : deblur}

        return encoder_dict


class SP_FPN_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.SP = nn.ModuleDict()
        self.DCN_extract = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.Run_sequence = nn.ModuleDict()
        self.FPN_level = 3
        self.num_bifpn = 0
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.num_sp_attention = 2
        self.motion_channel = 3 * (opt.per_pix_kernel **2) * opt.Recurrent_times

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4] #[opt.channel * 2 ** (i//2) for i in range(self.FPN_level)]
        in_channel_list = [3] + [opt.channel, opt.channel*2, opt.channel*4] #[opt.channel * 2 ** (i//2) for i in range(self.FPN_level-1)]
        stride_list = [1, 2, 2, 1, 1, 1]
        kernel = 3

        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            in_channel = in_channel_list[i-1]
            out_channel = out_channel_list[i-1]
            stride_length = stride_list[i-1]

            #if i in [1, 2]:
            #    att_mod_list = nn.ModuleDict()
            #    for j in range(self.num_sp_attention):
            #        att = f'att_{j+1}'
            #        att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, 2)
        
            #    self.sp_attention[level] = att_mod_list

            #elif i in [3, 4, 5, 6]:
            #    att_mod_list = nn.ModuleDict()
            #    for j in range(self.num_sp_attention):
            #        att = f'att_{j+1}'
            #        att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, 4)
        
            #    self.sp_attention[level] = att_mod_list


            self.SP[level] = StripPooling(opt, in_channel, strip_size = 1)
            self.DCN_extract[level] = nn.Sequential(
                DCN(in_channel, out_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),
                Residual_module(nn.Sequential(
                Base_Res_Block(opt, out_channel),
                DCN(out_channel, out_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2))))

            self.conv_up[level] = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding=1, stride = stride_length)
            self.Residual[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel), Base_Res_Block(opt, out_channel)))
            self.Run_sequence[level] = nn.Sequential(
                self.SP[level],
                self.DCN_extract[level],
                self.conv_up[level], self.Residual[level],
                DCN(out_channel, out_channel, kernel_size=(kernel,kernel), stride=1, padding = kernel//2),)
    
        
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            #if i % 2 == 1:
            #    self.bifpn[level] = BiFPN_attention_layer(opt, out_channel_list, stride_list)
            #else:
            self.bifpn[level] = BiFPN_layer(opt, out_channel_list, stride_list)



    def forward(self, x):
        # FPN goes up from Conv 1 - 4
        # Goes up
        FPN_info = {}
        for i in range(1,self.FPN_level+1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'
            if i == 1:
                FPN_info[fpn_feature] = self.Run_sequence[level](x)
            else:
                FPN_info[fpn_feature] = self.Run_sequence[level](FPN_info[last_fpn_feature])

        #for i in range(1, self.FPN_level + 1):
        #    level = f'level_{i}'
        #    fpn_feature = f'fpn_feature_lv{i}'
        #    for j in range(self.num_sp_attention):
        #        att = f'att_{j+1}'
        #        FPN_info[fpn_feature] = self.sp_attention[level][att](FPN_info[fpn_feature], motion)

        pred_bifpn = {}
        for i in range(1 ,4):#self.FPN_level+1):
            fpn_feature = f'fpn_feature_lv{i}'
            pred_bifpn[fpn_feature] = FPN_info[fpn_feature]    
        '''
        # Goes down (BiFPN part)
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            if i == 1:
                pred_bifpn = self.bifpn[level](FPN_info, motion)
            else:
                pred_bifpn = self.bifpn[level](pred_bifpn, motion)
        
        #return FPN_info, pred_bifpn
        '''

        encoder_dict = {'encode_feature' : pred_bifpn} #{'deblur' : deblur}

        return encoder_dict


class Patch_Unet_Encoder_1_2_4(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.reblur_attention = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel, opt.channel*2, opt.channel*4]

        ratio_list = [1, 2, 4]

        ## level 1
        channel = opt.channel
        self.Encoder_level1_Patch_FE = nn.Sequential(
                                        Guide_DCN_Block(opt, in_c = 3, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                        Base_Res_Block(opt, channel),
                                        Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, 1)
                                        )

        ## level 2
        channel = opt.channel * 2
        previous_channel = opt.channel
        self.Encoder_level2_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        Base_Res_Block(opt, channel),
                                        Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, 2)
                                        )

        ## level 3
        channel = opt.channel * 4
        previous_channel = opt.channel * 2
        self.Encoder_level3_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        Base_Res_Block(opt, channel),
                                        Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, 4)
                                        )     
                                          

    def get_image_level(self,x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]


    def append_reblur_filter(self, reblur_dict):
        self.reblur_filter_list.append(reblur_dict['reblur_filter'])
        return reblur_dict['attention_feature']


    def forward(self, x):
        self.get_image_level(x)
        self.feature_list = []
        self.reblur_filter_list = []
        Patch_Unet_dict = {}

        ## level 1
        H = x.size(2)
        W = x.size(3)
        feature_level = f'fpn_feature_lv{1}'
        feature_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
        feature_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
        feature_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
        feature_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
        feature_lv3_1 = self.Encoder_level1_Patch_FE(feature_lv3_1.contiguous())
        feature_lv3_2 = self.Encoder_level1_Patch_FE(feature_lv3_2.contiguous())
        feature_lv3_3 = self.Encoder_level1_Patch_FE(feature_lv3_3.contiguous())
        feature_lv3_4 = self.Encoder_level1_Patch_FE(feature_lv3_4.contiguous())
        feature_lv3_top = torch.cat((feature_lv3_1['attention_feature'], feature_lv3_2['attention_feature']), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3['attention_feature'], feature_lv3_4['attention_feature']), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv3
 
        feature_lv3_reblur_filter_top = torch.cat((feature_lv3_1['reblur_filter'], feature_lv3_2['reblur_filter']), 3)
        feature_lv3_reblur_filter_bot = torch.cat((feature_lv3_3['reblur_filter'], feature_lv3_4['reblur_filter']), 3)
        feature_lv3_reblur_filter = torch.cat((feature_lv3_reblur_filter_top, feature_lv3_reblur_filter_bot), 2)
        self.reblur_filter_list.append(feature_lv3_reblur_filter)

        ## level 2
        feature_level = f'fpn_feature_lv{2}'
        feature_lv2_1 = feature_lv3_top
        feature_lv2_2 = feature_lv3_bot
        feature_lv2_1 = self.Encoder_level2_Patch_FE(feature_lv2_1.contiguous())
        feature_lv2_2 = self.Encoder_level2_Patch_FE(feature_lv2_2.contiguous())
        feature_lv2 = torch.cat((feature_lv2_1['attention_feature'], feature_lv2_2['attention_feature']), 2)
        Patch_Unet_dict[feature_level] = feature_lv2

        feature_lv2_reblur_filter = torch.cat((feature_lv2_1['reblur_filter'], feature_lv2_2['reblur_filter']), 2)
        self.reblur_filter_list.append(feature_lv2_reblur_filter)


        ## level 3
        feature_level = f'fpn_feature_lv{3}'
        feature_lv1 = self.Encoder_level3_Patch_FE(feature_lv2)
        Patch_Unet_dict[feature_level] = feature_lv1['attention_feature']
        
        self.reblur_filter_list.append(feature_lv1['reblur_filter'])
        
        # out size : [batch, opt.channel * 4, H, W]
        encoder_dict = {'encode_feature' : Patch_Unet_dict, 'reblur_filter' : self.reblur_filter_list} #{'deblur' : deblur}

        return encoder_dict



class Patch_Unet_Encoder_1_2_4_f8(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.reblur_attention = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel, opt.channel*2, opt.channel*4]

        ratio_list = [1, 2, 4]

        ## level 1
        channel = opt.channel
        self.Encoder_level1_Patch_FE = nn.Sequential(
                                        Guide_DCN_Block(opt, in_c = 3, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                        #Base_Res_Block(opt, channel),
                                        #Base_Res_Block(opt, channel),
                                        #SP_reblur_attention(opt, channel, 1)
                                        )

        ## level 2
        channel = opt.channel * 2
        previous_channel = opt.channel
        self.Encoder_level2_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        #Base_Res_Block(opt, channel),
                                        #Base_Res_Block(opt, channel),
                                        #SP_reblur_attention(opt, channel, 2)
                                        )

        ## level 3
        channel = opt.channel * 4
        previous_channel = opt.channel * 2
        self.Encoder_level3_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        #Base_Res_Block(opt, channel),
                                        #Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, 4)
                                        )

        ## level 4
        channel = opt.channel * 4
        previous_channel = opt.channel * 4
        self.Encoder_level4_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                        Guide_DCN_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        #Base_Res_Block(opt, channel),
                                        #Base_Res_Block(opt, channel),
                                        SP_reblur_attention(opt, channel, 4)
                                        )     
                                          

    def get_image_level(self,x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        self.images_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
        self.images_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
        self.images_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
        self.images_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
        self.images_lv4_1 = self.images_lv3_1[:,:,0:int(H/4),:]
        self.images_lv4_2 = self.images_lv3_1[:,:,int(H/4):int(H/2),:]
        self.images_lv4_3 = self.images_lv3_2[:,:,0:int(H/4),:]
        self.images_lv4_4 = self.images_lv3_2[:,:,int(H/4):int(H/2),:]
        self.images_lv4_5 = self.images_lv3_3[:,:,0:int(H/4),:]
        self.images_lv4_6 = self.images_lv3_3[:,:,int(H/4):int(H/2),:]
        self.images_lv4_7 = self.images_lv3_4[:,:,0:int(H/4),:]
        self.images_lv4_8 = self.images_lv3_4[:,:,int(H/4):int(H/2),:]

    def append_reblur_filter(self, reblur_dict):
        self.reblur_filter_list.append(reblur_dict['reblur_filter'])
        return reblur_dict['attention_feature']


    def forward(self, x):
        self.get_image_level(x)
        self.feature_list = []
        self.reblur_filter_list = []
        Patch_Unet_dict = {}

        ## level 1
        H = x.size(2)
        W = x.size(3)
        feature_level = f'fpn_feature_lv{1}'
        feature_lv4_1 = self.Encoder_level1_Patch_FE(self.images_lv4_1.contiguous())
        feature_lv4_2 = self.Encoder_level1_Patch_FE(self.images_lv4_2.contiguous())
        feature_lv4_3 = self.Encoder_level1_Patch_FE(self.images_lv4_3.contiguous())
        feature_lv4_4 = self.Encoder_level1_Patch_FE(self.images_lv4_4.contiguous())
        feature_lv4_5 = self.Encoder_level1_Patch_FE(self.images_lv4_5.contiguous())
        feature_lv4_6 = self.Encoder_level1_Patch_FE(self.images_lv4_6.contiguous())
        feature_lv4_7 = self.Encoder_level1_Patch_FE(self.images_lv4_7.contiguous())
        feature_lv4_8 = self.Encoder_level1_Patch_FE(self.images_lv4_8.contiguous())

        feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv4

        '''
        feature_lv4_top_left = torch.cat((feature_lv4_1['attention_feature'], feature_lv4_2['attention_feature']), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3['attention_feature'], feature_lv4_4['attention_feature']), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5['attention_feature'], feature_lv4_6['attention_feature']), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7['attention_feature'], feature_lv4_8['attention_feature']), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv4

        feature_lv4_reblur_filter_top_left = torch.cat((feature_lv4_1['reblur_filter'], feature_lv4_2['reblur_filter']), 2)
        feature_lv4_reblur_filter_top_right = torch.cat((feature_lv4_3['reblur_filter'], feature_lv4_4['reblur_filter']), 2)
        feature_lv4_reblur_filter_bot_left = torch.cat((feature_lv4_5['reblur_filter'], feature_lv4_6['reblur_filter']), 2)
        feature_lv4_reblur_filter_bot_right = torch.cat((feature_lv4_7['reblur_filter'], feature_lv4_8['reblur_filter']), 2)
        feature_lv4_reblur_filter_top = torch.cat((feature_lv4_reblur_filter_top_left, feature_lv4_reblur_filter_top_right), 3)
        feature_lv4_reblur_filter_bot = torch.cat((feature_lv4_reblur_filter_bot_left, feature_lv4_reblur_filter_bot_right), 3)
        feature_lv4_reblur_filter = torch.cat((feature_lv4_reblur_filter_top, feature_lv4_reblur_filter_bot), 2)
        self.reblur_filter_list.append(feature_lv4_reblur_filter)
        '''

        ## level 2
        feature_level = f'fpn_feature_lv{2}'
        feature_lv3_1 = feature_lv4_top_left
        feature_lv3_2 = feature_lv4_top_right 
        feature_lv3_3 = feature_lv4_bot_left
        feature_lv3_4 = feature_lv4_bot_right 
        feature_lv3_1 = self.Encoder_level2_Patch_FE(feature_lv3_1.contiguous())
        feature_lv3_2 = self.Encoder_level2_Patch_FE(feature_lv3_2.contiguous())
        feature_lv3_3 = self.Encoder_level2_Patch_FE(feature_lv3_3.contiguous())
        feature_lv3_4 = self.Encoder_level2_Patch_FE(feature_lv3_4.contiguous())
        #'''
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv3

        '''
        feature_lv3_top = torch.cat((feature_lv3_1['attention_feature'], feature_lv3_2['attention_feature']), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3['attention_feature'], feature_lv3_4['attention_feature']), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv3
    
        feature_lv3_reblur_filter_top = torch.cat((feature_lv3_1['reblur_filter'], feature_lv3_2['reblur_filter']), 3)
        feature_lv3_reblur_filter_bot = torch.cat((feature_lv3_3['reblur_filter'], feature_lv3_4['reblur_filter']), 3)
        feature_lv3_reblur_filter = torch.cat((feature_lv3_reblur_filter_top, feature_lv3_reblur_filter_bot), 2)
        split = list(torch.split(feature_lv3_reblur_filter, 9, dim = 1))
        self.reblur_filter_list += split
        #'''

        ## level 3
        H = feature_lv3.size(2)
        W = feature_lv3.size(3)
        feature_level = f'fpn_feature_lv{3}'
        feature_lv2_1 = feature_lv3_top
        feature_lv2_2 = feature_lv3_bot
        feature_lv2_1 = self.Encoder_level3_Patch_FE(feature_lv2_1.contiguous())
        feature_lv2_2 = self.Encoder_level3_Patch_FE(feature_lv2_2.contiguous())
        '''
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
        Patch_Unet_dict[feature_level] = feature_lv2

        '''
        feature_lv2 = torch.cat((feature_lv2_1['attention_feature'], feature_lv2_2['attention_feature']), 2)
        Patch_Unet_dict[feature_level] = feature_lv2

        feature_lv2_reblur_filter = torch.cat((feature_lv2_1['reblur_filter'], feature_lv2_2['reblur_filter']), 2)
        split = list(torch.split(feature_lv2_reblur_filter, 9, dim = 1))
        self.reblur_filter_list += split
        #self.reblur_filter_list.append(feature_lv2_reblur_filter)
        #'''


        ## level 4
        feature_level = f'fpn_feature_lv{4}'
        feature_lv1 = self.Encoder_level4_Patch_FE(feature_lv2)
        '''
        Patch_Unet_dict[feature_level] = feature_lv1
        #self.reblur_filter_list.append(feature_lv1['reblur_filter'])
        '''
    
        Patch_Unet_dict[feature_level] = feature_lv1['attention_feature']
        split = list(torch.split(feature_lv1['reblur_filter'], 9, dim = 1))
        self.reblur_filter_list += split        
        #self.reblur_filter_list.append(feature_lv1['reblur_filter'])
        
        # out size : [batch, opt.channel * 4, H, W]
        encoder_dict = {'encode_feature' : Patch_Unet_dict, 'reblur_filter' : self.reblur_filter_list} #{'deblur' : deblur}

        return encoder_dict


class Patch_Unet_Encoder_1_2_4_8(nn.Module):
    '''
        by consecutive patch convolution solving motion blur from easy to hard.
        Start from the 4, which is to understand convolution is contructed in 4 different patch, but with same conv.. so on so fourth
    '''

    def __init__(self, opt):
        super().__init__()
        # Patch level feature extractor
        self.Patch_FE = nn.ModuleDict()
        self.non_linear_transform = nn.ModuleDict()
        self.get_motion = nn.ModuleDict()
        self.reblur_attention = nn.ModuleDict()

        self.patch_level = 3
        channel_list = [opt.channel, opt.channel*2, opt.channel*4]

        ratio_list = [1, 2, 4]

        ## level 1
        channel = opt.channel
        self.Encoder_level1_Patch_FE = nn.Sequential(
                                        Component_Block(opt, in_c = 3, out_c = channel, stride_len = 1),
                                        #SP_reblur_attention(opt, channel, 1)
                                        )

        ## level 2
        channel = opt.channel * 2
        previous_channel = opt.channel
        self.Encoder_level2_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        #SP_reblur_attention(opt, channel, 2)
                                        )

        ## level 3
        channel = opt.channel * 4
        previous_channel = opt.channel * 2
        self.Encoder_level3_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 2, dilation=1),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        SP_reblur_attention(opt, channel, 4)
                                        )

        ## level 4
        channel = opt.channel * 4
        previous_channel = opt.channel * 4
        self.Encoder_level4_Patch_FE = nn.Sequential(
                                        nn.Conv2d(previous_channel, channel, kernel_size = 3, padding = 1, stride = 1, dilation=1),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        SP_reblur_attention(opt, channel, 4)
                                        )     
                                          

    def get_image_level(self,x):
        H = x.size(2)
        W = x.size(3)
        images_lv1 = x
        self.images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        self.images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        self.images_lv3_1 = self.images_lv2_1[:,:,:,0:int(W/2)]
        self.images_lv3_2 = self.images_lv2_1[:,:,:,int(W/2):W]
        self.images_lv3_3 = self.images_lv2_2[:,:,:,0:int(W/2)]
        self.images_lv3_4 = self.images_lv2_2[:,:,:,int(W/2):W]
        self.images_lv4_1 = self.images_lv3_1[:,:,0:int(H/4),:]
        self.images_lv4_2 = self.images_lv3_1[:,:,int(H/4):int(H/2),:]
        self.images_lv4_3 = self.images_lv3_2[:,:,0:int(H/4),:]
        self.images_lv4_4 = self.images_lv3_2[:,:,int(H/4):int(H/2),:]
        self.images_lv4_5 = self.images_lv3_3[:,:,0:int(H/4),:]
        self.images_lv4_6 = self.images_lv3_3[:,:,int(H/4):int(H/2),:]
        self.images_lv4_7 = self.images_lv3_4[:,:,0:int(H/4),:]
        self.images_lv4_8 = self.images_lv3_4[:,:,int(H/4):int(H/2),:]

    def append_reblur_filter(self, reblur_dict):
        self.reblur_filter_list.append(reblur_dict['reblur_filter'])
        return reblur_dict['attention_feature']


    def forward(self, x):
        self.get_image_level(x)
        self.feature_list = []
        self.reblur_filter_list = []
        Patch_Unet_dict = {}

        ## level 1
        H = x.size(2)
        W = x.size(3)
        feature_level = f'fpn_feature_lv{1}'
        feature_lv4_1 = self.Encoder_level1_Patch_FE(self.images_lv4_1.contiguous())
        feature_lv4_2 = self.Encoder_level1_Patch_FE(self.images_lv4_2.contiguous())
        feature_lv4_3 = self.Encoder_level1_Patch_FE(self.images_lv4_3.contiguous())
        feature_lv4_4 = self.Encoder_level1_Patch_FE(self.images_lv4_4.contiguous())
        feature_lv4_5 = self.Encoder_level1_Patch_FE(self.images_lv4_5.contiguous())
        feature_lv4_6 = self.Encoder_level1_Patch_FE(self.images_lv4_6.contiguous())
        feature_lv4_7 = self.Encoder_level1_Patch_FE(self.images_lv4_7.contiguous())
        feature_lv4_8 = self.Encoder_level1_Patch_FE(self.images_lv4_8.contiguous())

        feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv4

        '''
        feature_lv4_top_left = torch.cat((feature_lv4_1['attention_feature'], feature_lv4_2['attention_feature']), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3['attention_feature'], feature_lv4_4['attention_feature']), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5['attention_feature'], feature_lv4_6['attention_feature']), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7['attention_feature'], feature_lv4_8['attention_feature']), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv4

        feature_lv4_reblur_filter_top_left = torch.cat((feature_lv4_1['reblur_filter'], feature_lv4_2['reblur_filter']), 2)
        feature_lv4_reblur_filter_top_right = torch.cat((feature_lv4_3['reblur_filter'], feature_lv4_4['reblur_filter']), 2)
        feature_lv4_reblur_filter_bot_left = torch.cat((feature_lv4_5['reblur_filter'], feature_lv4_6['reblur_filter']), 2)
        feature_lv4_reblur_filter_bot_right = torch.cat((feature_lv4_7['reblur_filter'], feature_lv4_8['reblur_filter']), 2)
        feature_lv4_reblur_filter_top = torch.cat((feature_lv4_reblur_filter_top_left, feature_lv4_reblur_filter_top_right), 3)
        feature_lv4_reblur_filter_bot = torch.cat((feature_lv4_reblur_filter_bot_left, feature_lv4_reblur_filter_bot_right), 3)
        feature_lv4_reblur_filter = torch.cat((feature_lv4_reblur_filter_top, feature_lv4_reblur_filter_bot), 2)
        self.reblur_filter_list.append(feature_lv4_reblur_filter)
        '''

        ## level 2
        feature_level = f'fpn_feature_lv{2}'
        feature_lv3_1 = feature_lv4_top_left
        feature_lv3_2 = feature_lv4_top_right 
        feature_lv3_3 = feature_lv4_bot_left
        feature_lv3_4 = feature_lv4_bot_right 
        feature_lv3_1 = self.Encoder_level2_Patch_FE(feature_lv3_1.contiguous())
        feature_lv3_2 = self.Encoder_level2_Patch_FE(feature_lv3_2.contiguous())
        feature_lv3_3 = self.Encoder_level2_Patch_FE(feature_lv3_3.contiguous())
        feature_lv3_4 = self.Encoder_level2_Patch_FE(feature_lv3_4.contiguous())
        #'''
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv3

        '''
        feature_lv3_top = torch.cat((feature_lv3_1['attention_feature'], feature_lv3_2['attention_feature']), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3['attention_feature'], feature_lv3_4['attention_feature']), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        Patch_Unet_dict[feature_level] = feature_lv3
    
        feature_lv3_reblur_filter_top = torch.cat((feature_lv3_1['reblur_filter'], feature_lv3_2['reblur_filter']), 3)
        feature_lv3_reblur_filter_bot = torch.cat((feature_lv3_3['reblur_filter'], feature_lv3_4['reblur_filter']), 3)
        feature_lv3_reblur_filter = torch.cat((feature_lv3_reblur_filter_top, feature_lv3_reblur_filter_bot), 2)
        split = list(torch.split(feature_lv3_reblur_filter, 9, dim = 1))
        self.reblur_filter_list += split
        #'''

        ## level 3
        H = feature_lv3.size(2)
        W = feature_lv3.size(3)
        feature_level = f'fpn_feature_lv{3}'
        feature_lv2_1 = feature_lv3_top
        feature_lv2_2 = feature_lv3_bot
        feature_lv2_1 = self.Encoder_level3_Patch_FE(feature_lv2_1.contiguous())
        feature_lv2_2 = self.Encoder_level3_Patch_FE(feature_lv2_2.contiguous())
        '''
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
        Patch_Unet_dict[feature_level] = feature_lv2

        '''
        feature_lv2 = torch.cat((feature_lv2_1['attention_feature'], feature_lv2_2['attention_feature']), 2)
        Patch_Unet_dict[feature_level] = feature_lv2

        feature_lv2_reblur_filter = torch.cat((feature_lv2_1['reblur_filter'], feature_lv2_2['reblur_filter']), 2)
        split = list(torch.split(feature_lv2_reblur_filter, 9, dim = 1))
        self.reblur_filter_list += split
        #self.reblur_filter_list.append(feature_lv2_reblur_filter)
        #'''


        ## level 4
        feature_level = f'fpn_feature_lv{4}'
        feature_lv1 = self.Encoder_level4_Patch_FE(feature_lv2)
        '''
        Patch_Unet_dict[feature_level] = feature_lv1
        #self.reblur_filter_list.append(feature_lv1['reblur_filter'])
        '''
    
        Patch_Unet_dict[feature_level] = feature_lv1['attention_feature']
        split = list(torch.split(feature_lv1['reblur_filter'], 9, dim = 1))
        self.reblur_filter_list += split        
        #self.reblur_filter_list.append(feature_lv1['reblur_filter'])
        
        # out size : [batch, opt.channel * 4, H, W]
        encoder_dict = {'encode_feature' : Patch_Unet_dict, 'reblur_filter' : self.reblur_filter_list} #{'deblur' : deblur}

        return encoder_dict