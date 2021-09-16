import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_modules.component_block import *
from sub_modules.attention_zoo import *
from sub_modules.bifpn_block import *


class SP_FPN_decoder(nn.Module):
    """
        Reference : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/15403b5371a64defb2a7c74e162c6e880a7f462c/efficientdet/model.py#L55
    """
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.conv_down = nn.ModuleDict()
        self.conv_pred = nn.ModuleDict()
        self.prediction_weight = {}
        self.FPN_level = 6
        self.num_bifpn = 0
        self.epsilon = 1e-4
        self.leaky = nn.LeakyReLU(0.2, True)
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.num_sp_attention = 2
        self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
        self.motion_channel = 3 * (opt.per_pix_kernel **2 ) * opt.Recurrent_times

        out_channel_list = [opt.channel * 2 ** (i//2) for i in range(self.FPN_level)]
        self.a = out_channel_list
        in_channel_list = [3] + [opt.channel * 2 ** (i//2) for i in range(self.FPN_level-1)]
        stride_list = [2, 2, 1, 1, 1, 1]

        for i in range(1, self.FPN_level + 1):
            level = f'level_{i}'

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

            self.conv_pred[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),))


            if i != 1:
                self.Residual[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),)) 
                if stride_list[i-2] == 2:
                    self.conv_down[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                            nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-2]), Base_Res_Block(opt, out_channel_list[i-2]),)),
                                            Base_Res_Block(opt, out_channel_list[i-2]),
                                            self.leaky)         
                else:
                    self.conv_down[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            Base_Res_Block(opt, out_channel_list[i-2]),
                                            self.leaky)    
            
            else:
                self.Residual[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                                    Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),)),
                                                    Base_Res_Block(opt, out_channel_list[i-1]),
                                                    nn.Conv2d(out_channel_list[i-1], in_channel_list[i-1], kernel_size=3, padding = 1, dilation = 1),
                                                    self.leaky,)
                #self.final_predict = Base_Res_Block(opt, in_channel_list[i-1])

            if i in [2, 3, 4, 5]:
                self.prediction_weight[level] = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True).to(self.isgpu)
            else:
                self.prediction_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu) 
        
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            self.bifpn[level] = BiFPN_layer(opt, out_channel_list, stride_list)



    def forward(self, FPN_info, pred_bifpn, motion):
        """
        illustration of a SP_FPN_decoder, each final will add with weighted sum of residual path

            P6_0 -------------------------> P6_2 --------> P6_final
               |-------------|                ↑               |
                             ↓                |               ↓
            P5_0 ---------> P5_1 ---------> P5_2 --------> P5_final
               |-------------|--------------↑ ↑               |
                             ↓                |               ↓
            P4_0 ---------> P4_1 ---------> P4_2 --------> P4_final
               |-------------|--------------↑ ↑               |
                             ↓                |               ↓
            P3_0 ---------> P3_1 ---------> P3_2 --------> P3_final
               |-------------|--------------↑ ↑               |
                             |--------------↓ |               ↓
            P2_0 -------------------------> P2_2 --------> P2_final
                                                              |
                                                              ↓          
                                                           P1_final    
        """
        
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            pred_bifpn = self.bifpn[level](pred_bifpn, motion)

        #for i in range(1, self.FPN_level + 1):
        #    level = f'level_{i}'
        #    fpn_feature = f'fpn_feature_lv{i}'
        #    for j in range(self.num_sp_attention):
        #        att = f'att_{j+1}'
        #        FPN_info[fpn_feature] = self.sp_attention[level][att](FPN_info[fpn_feature], motion)

        predict_FPN_info = {}
        for i in range(self.FPN_level, 0, -1):
            level = f'level_{i}'
            top_level = f'level_{i+1}'
            fpn_feature = f'fpn_feature_lv{i}'
            top_fpn_feature = f'fpn_feature_lv{i+1}'   

            if i in [2, 3, 4, 5]:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_residual_path = FPN_info[fpn_feature]
                from_res_block = self.Residual[level](pred_bifpn[fpn_feature])
                from_top = self.conv_down[top_level](predict_FPN_info[top_fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_res_block + norm_w[1] * from_residual_path + norm_w[2] * from_top)
                #predict_FPN_info[fpn_feature] = from_res_block + from_residual_path + from_top

            else:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                if i == 6:
                    from_residual_path = FPN_info[fpn_feature]
                    from_res_block = self.Residual[level](pred_bifpn[fpn_feature])
                    predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_res_block + norm_w[1] * from_residual_path)
                    #predict_FPN_info[fpn_feature] = from_res_block + from_residual_path

                else:
                    from_residual_path = FPN_info[fpn_feature]
                    from_res_block = pred_bifpn[top_fpn_feature]
                    fusion_feature = self.Residual[level](self.conv_pred[level](norm_w[0] * from_res_block + norm_w[1] * from_residual_path))
                    #fusion_feature = self.Residual[level](from_res_block + from_residual_path)
                    predict_FPN_info[fpn_feature] = fusion_feature
            

        return predict_FPN_info[fpn_feature]



class SP_FPN_DCN_decoder(nn.Module):
    """
        Reference : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/15403b5371a64defb2a7c74e162c6e880a7f462c/efficientdet/model.py#L55
    """
    def __init__(self, opt):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.conv_down = nn.ModuleDict()
        self.conv_pred = nn.ModuleDict()
        self.prediction_weight = {}
        self.FPN_level = 6
        self.num_bifpn = 0
        self.epsilon = 1e-4
        self.leaky = nn.LeakyReLU(0.2, True)
        self.bifpn = nn.ModuleDict()
        self.sp_attention = nn.ModuleDict()
        self.num_sp_attention = 2
        self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
        self.motion_channel = 3 * (opt.per_pix_kernel **2 ) * opt.Recurrent_times

        out_channel_list = [opt.channel * 2 ** (i//2) for i in range(self.FPN_level)]
        self.a = out_channel_list
        in_channel_list = [3] + [opt.channel * 2 ** (i//2) for i in range(self.FPN_level-1)]
        stride_list = [2, 2, 1, 1, 1, 1]

        for i in range(1, self.FPN_level + 1):
            level = f'level_{i}'

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

            self.conv_pred[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),))


            if i != 1:
                self.Residual[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),)) 
                if stride_list[i-2] == 2:
                    self.conv_down[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                            nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-2]), Base_Res_Block(opt, out_channel_list[i-2]),)),
                                            Base_Res_Block(opt, out_channel_list[i-2]),
                                            self.leaky)         
                else:
                    self.conv_down[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            Base_Res_Block(opt, out_channel_list[i-2]),
                                            self.leaky)    
            
            else:
                self.Residual[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                                    Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),)),
                                                    Base_Res_Block(opt, out_channel_list[i-1]),
                                                    nn.Conv2d(out_channel_list[i-1], in_channel_list[i-1], kernel_size=3, padding = 1, dilation = 1),
                                                    self.leaky,)
                #self.final_predict = Base_Res_Block(opt, in_channel_list[i-1])

            if i in [2, 3, 4, 5]:
                self.prediction_weight[level] = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True).to(self.isgpu)
            else:
                self.prediction_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu) 
        
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            self.bifpn[level] = BiFPN_layer(opt, out_channel_list, stride_list)



    def forward(self, FPN_info, pred_bifpn, motion):
        """
        illustration of a SP_FPN_decoder, each final will add with weighted sum of residual path

            P6_0 -------------------------> P6_2 --------> P6_final
               |-------------|                ↑               |
                             ↓                |               ↓
            P5_0 ---------> P5_1 ---------> P5_2 --------> P5_final
               |-------------|--------------↑ ↑               |
                             ↓                |               ↓
            P4_0 ---------> P4_1 ---------> P4_2 --------> P4_final
               |-------------|--------------↑ ↑               |
                             ↓                |               ↓
            P3_0 ---------> P3_1 ---------> P3_2 --------> P3_final
               |-------------|--------------↑ ↑               |
                             |--------------↓ |               ↓
            P2_0 -------------------------> P2_2 --------> P2_final
                                                              |
                                                              ↓          
                                                           P1_final    
        """
        
        for i in range(1, self.num_bifpn + 1):
            level = f'level_{i}' 
            pred_bifpn = self.bifpn[level](pred_bifpn, motion)

        #for i in range(1, self.FPN_level + 1):
        #    level = f'level_{i}'
        #    fpn_feature = f'fpn_feature_lv{i}'
        #    for j in range(self.num_sp_attention):
        #        att = f'att_{j+1}'
        #        FPN_info[fpn_feature] = self.sp_attention[level][att](FPN_info[fpn_feature], motion)

        predict_FPN_info = {}
        for i in range(self.FPN_level, 0, -1):
            level = f'level_{i}'
            top_level = f'level_{i+1}'
            fpn_feature = f'fpn_feature_lv{i}'
            top_fpn_feature = f'fpn_feature_lv{i+1}'   

            if i in [2, 3, 4, 5]:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_residual_path = FPN_info[fpn_feature]
                from_res_block = self.Residual[level](pred_bifpn[fpn_feature])
                from_top = self.conv_down[top_level](predict_FPN_info[top_fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_res_block + norm_w[1] * from_residual_path + norm_w[2] * from_top)
                #predict_FPN_info[fpn_feature] = from_res_block + from_residual_path + from_top

            else:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                if i == 6:
                    from_residual_path = FPN_info[fpn_feature]
                    from_res_block = self.Residual[level](pred_bifpn[fpn_feature])
                    predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_res_block + norm_w[1] * from_residual_path)
                    #predict_FPN_info[fpn_feature] = from_res_block + from_residual_path

                else:
                    from_residual_path = FPN_info[fpn_feature]
                    from_res_block = pred_bifpn[top_fpn_feature]
                    fusion_feature = self.Residual[level](self.conv_pred[level](norm_w[0] * from_res_block + norm_w[1] * from_residual_path))
                    #fusion_feature = self.Residual[level](from_res_block + from_residual_path)
                    predict_FPN_info[fpn_feature] = fusion_feature
            

        return predict_FPN_info[fpn_feature]
                
# 1-2-4
class BiFPN_Decoder(nn.Module):
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
        self.epsilon = 1e-8
        self.prediction_weight = {}
        self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
        self.pac = nn.ModuleDict()
        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.FPN_level = 4

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        in_channel_list = [3, opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [1 ,2 ,4 ,4]
        stride_list = [1, 2, 2, 1] # 256, 128, 64, 64
        previous_ratio = 4

        for i in range(self.FPN_level, 0, -1):
            deconv = f'Deconv{i}'
            channel = in_channel_list[i]
            previous_channel = in_channel_list[i-1]
            ratio = ratio_list[i-1]
            if ratio == 4:
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))

            if i != 1:
                self.pac[deconv] = nn.Conv2d(channel, previous_channel, kernel_size=3, stride=1, padding=1)
                self.sft_pac[deconv] = SFT_DCN(previous_channel, self.motion_channel, ratio)

            else:
                self.pac[deconv] = nn.Conv2d(channel, previous_channel, kernel_size=3, stride=1, padding=1)

            self.sft_lv1[deconv] = SFT_DCN(channel, self.motion_channel, ratio)
            self.sft_lv2[deconv] = SFT_DCN(channel, self.motion_channel, ratio)

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)

            if i != 1 and i != self.FPN_level:
                self.prediction_weight[deconv] = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True).to(self.isgpu)
            else:
                self.prediction_weight[deconv] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self, FPN_info, pred_bifpn, motion):
        """
        illustration of a SP_FPN_decoder, each final will add with weighted sum of residual path

            P4_0 -------------------------> P4_2 --------> P4_final
               |-------------|                ↑               |
                             ↓                |               ↓
            P3_0 ---------> P3_1 ---------> P3_2 --------> P3_final
               |-------------|--------------↑ ↑               |
                             |--------------↓ |               ↓
            P2_0 -------------------------> P2_2 --------> P2_final
                                                              |
                                                              ↓          
            P1_origin ---------------------------------->  P1_final    --> P1 out + residual = deblur
        """

        self.feature_list = []
        predict_FPN_info = {}

        pac_result = None
        for i in range(self.FPN_level,0,-1):
            deconv = f'Deconv{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}' 
            top_fpn_feature = f'fpn_feature_lv{i+1}'  

            if i != self.FPN_level:
                x = predict_FPN_info[top_fpn_feature]
                residual_feature = predict_FPN_info[top_fpn_feature]

            else:
                x = pred_bifpn[last_fpn_feature]
                residual_feature = pred_bifpn[last_fpn_feature]

            x = self.conv_up[deconv](x)
            x = self.SFT_fusion(x, motion, self.non_linear_transform_lv1[deconv], self.sft_lv1[deconv])
            styled_feature = self.SFT_fusion(x, motion, self.non_linear_transform_lv2[deconv], self.sft_lv2[deconv])

            #w = (self.prediction_weight[deconv])
            #norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

            if i != 1 and i != self.FPN_level:
                from_residual = FPN_info[fpn_feature]
                from_lateral = pred_bifpn[fpn_feature]
                from_top = styled_feature
                #weight_feature = self.pac[deconv](norm_w[0] * from_lateral + norm_w[1] * from_residual + norm_w[2] * from_top)
                weight_feature = self.pac[deconv](from_lateral + from_residual + from_top)  
                predict_FPN_info[fpn_feature] =  self.sft_pac[deconv](weight_feature, motion)

            elif i == self.FPN_level:
                from_residual = FPN_info[fpn_feature]
                from_lateral = styled_feature
                #weight_feature = self.pac[deconv](norm_w[0] * from_lateral + norm_w[1] * from_residual) 
                weight_feature = self.pac[deconv](from_lateral + from_residual)  
                predict_FPN_info[fpn_feature] =  self.sft_pac[deconv](weight_feature, motion)

            else:
                from_residual = FPN_info[fpn_feature]
                from_top = styled_feature
                #predict_FPN_info[fpn_feature] = self.pac[deconv](norm_w[0] * from_top + norm_w[1] * from_residual) 
                predict_FPN_info[fpn_feature] = self.pac[deconv](from_top + from_residual)                                
            
            self.feature_list.append(predict_FPN_info[fpn_feature])

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return predict_FPN_info[fpn_feature]




    # 1-2-4
class FPN_Decoder_SFT_fusion(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac_lv1 = nn.ModuleDict()
        self.sft_pac_lv2 = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 3
        self.num_predict_ORB = 3
        
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

                self.sft_pac_lv1[deconv] = SP_motion_attention(opt, channel//2, self.motion_channel, 2**(i - 2))
                self.sft_pac_lv2[deconv] = SP_motion_attention(opt, channel//2, self.motion_channel, 2**(i - 2))
                #SFT_DCN(channel//2, self.motion_channel, 2**(i - 2))

            if i != 1:
                self.pac[deconv] = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    self.conv_ORB[level] = Base_Res_Block(opt, channel)
                    self.attention_ORB[level] = SP_motion_attention(opt, channel, self.motion_channel, 1)

                for j in range(self.num_predict_ORB):
                    level = f'Deconv{j + 1}'
                    self.predict_ORB[level] = Base_Res_Block(opt, 3)
                    self.attention_predict_ORB[level] = SP_motion_attention(opt, 3, self.motion_channel, 1)

            self.sft_lv1[deconv] = SP_motion_attention(opt, channel, self.motion_channel, 2**(i - 1)) #SFT_DCN(channel, self.motion_channel, 2**(i - 1))
            self.sft_lv2[deconv] = SP_motion_attention(opt, channel, self.motion_channel, 2**(i - 1)) #SFT_DCN(channel, self.motion_channel, 2**(i - 1))

            self.non_linear_transform_lv1[deconv] = get_norm_layer(opt.Norm, channel)
            self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)
        
        
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self, x, motion_feature):
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
            
            if i != 1:
                pac_result = self.pac[deconv](styled_feature)
                pac_result = self.sft_pac_lv1[deconv](pac_result, motion_feature)
                pac_result = self.sft_pac_lv2[deconv](pac_result, motion_feature)
                
            else:
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    pac_result = self.conv_ORB[level](pac_result)
                    pac_result = self.attention_ORB[level](pac_result, motion_feature)
                pac_result = self.pac[deconv](pac_result)
                for j in range(self.num_predict_ORB):
                    level = f'Deconv{j + 1}'
                    pac_result = self.predict_ORB[level](pac_result)
                    pac_result = self.attention_predict_ORB[level](pac_result, motion_feature)

            self.feature_list.append(pac_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result


# 1-2-4
class PAN_Decoder(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.sft_lv1 = nn.ModuleDict()
        self.sft_lv2 = nn.ModuleDict()
        self.sft_pac_lv1 = nn.ModuleDict()
        self.sft_pac_lv2 = nn.ModuleDict()
        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.pac = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 0
        self.num_predict_ORB = 2
        
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

                self.sft_pac_lv1[deconv] = SFT_DCN(opt, channel//2, self.motion_channel, 2**(i - 2))
                #self.sft_pac_lv2[deconv] = SP_motion_attention(opt, channel//2, self.motion_channel, 2**(i - 2))
                #SFT_DCN(channel//2, self.motion_channel, 2**(i - 2))

            if i != 1:
                self.pac[deconv] = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.pac[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    self.conv_ORB[level] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    Res_Block('leaky', channel),
                                                    )
                    self.attention_ORB[level] = motion_Dynamic_filter(opt, channel, self.motion_channel, 1)

                self.ORB = nn.Sequential(
                                        Res_Block('leaky', channel),
                                        Res_Block('leaky', channel),
                                        Res_Block('leaky', channel),
                                        )

                for j in range(self.num_predict_ORB):
                    level = f'Deconv{j + 1}'
                    self.predict_ORB[level] = Base_Res_Block(opt, 3)
                    #self.attention_predict_ORB[level] = SP_motion_attention(opt, 3, self.motion_channel, 1)

            self.sft_lv1[deconv] = SFT_DCN(opt, channel, self.motion_channel, 2**(i - 1))
            self.sft_lv2[deconv] = SFT_DCN(opt, channel, self.motion_channel, 2**(i - 1))

            self.non_linear_transform_lv1[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    Res_Block('leaky', channel),
                                                    )
            self.non_linear_transform_lv2[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    Res_Block('leaky', channel),
                                                    )
            #self.non_linear_transform_lv2[deconv] = get_norm_layer(opt.Norm, channel)
        
        
    
    def SFT_fusion(self, x, motion_feature, network_transform, sft_transform):
        feature = x
        feature = sft_transform(feature, motion_feature)
        return x + network_transform(feature)
        
    def forward(self, x, motion_feature):
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
            
            if i != 1:
                pac_result = self.pac[deconv](styled_feature)
                pac_result = self.sft_pac_lv1[deconv](pac_result, motion_feature)
                #pac_result = self.sft_pac_lv2[deconv](pac_result, motion_feature)
                
            else:
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    pac_result = self.attention_ORB[level](pac_result, motion_feature)
                    pac_result = self.conv_ORB[level](pac_result)
                    
                pac_result = self.pac[deconv](self.ORB(pac_result))

                for j in range(self.num_predict_ORB):
                    level = f'Deconv{j + 1}'
                    pac_result = self.predict_ORB[level](pac_result)
                    #pac_result = self.attention_predict_ORB[level](pac_result, motion_feature)

            self.feature_list.append(pac_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 2, H, W)    
        return pac_result



# 1-2-4
class FPN_Decoder_SFT(nn.Module):
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
        self.FPN_level = 3
        

        for i in range(self.FPN_level,0,-1):
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
        
    def forward(self, pred_bifpn, motion_feature):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.feature_list = []
        pac_result = None
        x = pred_bifpn#self.ASPP(pred_bifpn)#[f'fpn_feature_lv{self.FPN_level}']
        
        for i in range(self.FPN_level, 0, -1):
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



# 1-2-4
class PAN_self_reblur_Decoder(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.reblur_attention_lv1 = nn.ModuleDict()
        self.reblur_attention_lv2 = nn.ModuleDict()
        self.reblur_attention_lv3 = nn.ModuleDict()

        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_reducde_channel = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 2
        self.num_predict_ORB = 2
        
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 1:
                channel = opt.channel
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = opt.channel*2*(i-1)
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))

                self.reblur_attention_lv3[deconv] = SP_reblur_attention(opt, channel//2, 2**(i - 2))

            if i != 1:
                self.conv_reducde_channel[deconv] = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_reducde_channel[deconv] = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    self.conv_ORB[level] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    #Res_Block('leaky', channel),
                                                    )
                    self.attention_ORB[level] = SP_reblur_attention(opt, channel, 1)

                self.ORB = nn.Sequential(
                                        Res_Block('leaky', channel),
                                        #Res_Block('leaky', channel),
                                        #Res_Block('leaky', channel),
                                        )

                for j in range(self.num_predict_ORB):
                    level = f'Deconv{j + 1}'
                    self.predict_ORB[level] = Base_Res_Block(opt, 3)
                    self.attention_predict_ORB[level] = SP_reblur_attention(opt, 3, 1)

            self.reblur_attention_lv1[deconv] = SP_reblur_attention(opt, channel, 2**(i - 1))
            self.reblur_attention_lv2[deconv] = SP_reblur_attention(opt, channel, 2**(i - 1))

            self.non_linear_transform_lv1[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    #Res_Block('leaky', channel),
                                                    )
            self.non_linear_transform_lv2[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    #Res_Block('leaky', channel),
                                                    )
        
        
    
    def reblur_fusion(self, x, network_transform, reblur_transform):
        feature = x
        feature_dict = reblur_transform(feature)

        self.reblur_filter_list.append(feature_dict['reblur_filter'])

        return x + network_transform(feature_dict['attention_feature'])
        
    def forward(self, x):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.feature_list = []
        self.reblur_filter_list = []
        decode_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = decode_result
                residual_feature = x
    
            x = self.reblur_fusion(x, self.non_linear_transform_lv1[deconv], self.reblur_attention_lv1[deconv])
            x = self.reblur_fusion(x, self.non_linear_transform_lv2[deconv], self.reblur_attention_lv2[deconv])
            styled_feature = self.conv_up[deconv](x + residual_feature)
            
            if i != 1:
                decode_result = self.conv_reducde_channel[deconv](styled_feature)
                decode_result = self.reblur_attention_lv3[deconv](decode_result)
                self.reblur_filter_list.append(decode_result['reblur_filter'])
                decode_result = decode_result['attention_feature']
                
            else:
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    decode_result = self.attention_ORB[level](decode_result)
                    self.reblur_filter_list.append(decode_result['reblur_filter'])
                    decode_result = decode_result['attention_feature']
                    decode_result = self.conv_ORB[level](decode_result)
                    
                decode_result = self.conv_reducde_channel[deconv](self.ORB(decode_result))

                for j in range(self.num_predict_ORB):
                    level = f'Deconv{j + 1}'
                    decode_result = self.attention_predict_ORB[level](decode_result)
                    self.reblur_filter_list.append(decode_result['reblur_filter'])
                    decode_result = decode_result['attention_feature']
                    decode_result = self.predict_ORB[level](decode_result)

            self.feature_list.append(decode_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}

        return decode_dict



# 1-2-4
class Patch_Unet_Decodger(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.reblur_attention_lv1 = nn.ModuleDict()
        self.reblur_attention_lv2 = nn.ModuleDict()
        self.reblur_attention_lv3 = nn.ModuleDict()

        self.non_linear_transform_lv1 = nn.ModuleDict()
        self.non_linear_transform_lv2 = nn.ModuleDict()
        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.unet_fusion = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 5

        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            ratio = ratio_list[i-1]
            if i == 3:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))

            if i != 1:
                self.reblur_attention_lv3[deconv] = SP_reblur_attention(opt, channel//2, ratio)

            if i == 3:
                self.unet_fusion[deconv] = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)

            elif i != 1:
                self.conv_lateral[deconv] = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
                self.unet_fusion[deconv] = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)  

            else:
                self.conv_lateral[deconv] = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
                self.unet_fusion[deconv] = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    self.conv_ORB[level] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    )
                    self.attention_ORB[level] = SP_reblur_attention(opt, channel, 1)

                self.ORB = nn.Sequential(
                                        Res_Block('leaky', channel),
                                        )

                self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

            self.reblur_attention_lv1[deconv] = SP_reblur_attention(opt, channel, ratio)
            self.reblur_attention_lv2[deconv] = SP_reblur_attention(opt, channel, ratio)

            self.non_linear_transform_lv1[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    )
            self.non_linear_transform_lv2[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    )
        
        
    
    def reblur_fusion(self, x, network_transform, reblur_transform):
        feature = x
        feature_dict = reblur_transform(feature)

        self.reblur_filter_list.append(feature_dict['reblur_filter'])

        return x + network_transform(feature_dict['attention_feature'])
        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####

        self.feature_list = []
        self.reblur_filter_list = []
        x = encode[f'fpn_feature_lv3']
        decode_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            level = f'fpn_feature_lv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = decode_result
                residual_feature = x

            x = self.conv_up[deconv](x)
            x = self.reblur_fusion(x, self.non_linear_transform_lv1[deconv], self.reblur_attention_lv1[deconv])
            styled_feature = self.reblur_fusion(x, self.non_linear_transform_lv2[deconv], self.reblur_attention_lv2[deconv])
            
            if i == 3:
                decode_result = self.unet_fusion[deconv](styled_feature)
                decode_result = self.reblur_attention_lv3[deconv](decode_result)
                self.reblur_filter_list.append(decode_result['reblur_filter'])
                decode_result = decode_result['attention_feature']

            elif i != 1:
                decode_result = self.unet_fusion[deconv](styled_feature + self.conv_lateral[deconv](encode[level]))
                decode_result = self.reblur_attention_lv3[deconv](decode_result)
                self.reblur_filter_list.append(decode_result['reblur_filter'])
                decode_result = decode_result['attention_feature']
                
            else:
                decode_result = self.unet_fusion[deconv](styled_feature + self.conv_lateral[deconv](encode[level]))

                for j in range(self.num_ORB):
                    deconv_level = f'Deconv{j + 1}'
                    decode_result = self.attention_ORB[deconv_level](decode_result)
                    self.reblur_filter_list.append(decode_result['reblur_filter'])
                    decode_result = decode_result['attention_feature']
                    decode_result = self.conv_ORB[deconv_level](decode_result)
                    
                decode_result = self.predict(self.ORB(decode_result))

            self.feature_list.append(decode_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}

        return decode_dict



# 1-2-4
class Patch_Unet_Decoderf(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.reblur_attention_lv1 = nn.ModuleDict()
        self.reblur_attention_lv2 = nn.ModuleDict()
        self.reblur_attention_lv3 = nn.ModuleDict()

        self.non_linear_transform = nn.ModuleDict()

        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.unet_fusion = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 5

        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            ratio = ratio_list[i-1]
            if i == 3:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))

            if i == 3:
                self.unet_fusion[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
                                            Res_Block('leaky', channel//2),
                                            Res_Block('leaky', channel//2),
                                            )

            elif i != 1:
                self.conv_lateral[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            #Res_Block('leaky', channel),
                                            #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                            #channel_attention(channel),
                                            )  

                self.unet_fusion[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
                                            Res_Block('leaky', channel//2),
                                            Res_Block('leaky', channel//2),
                                            )
            else:
                self.conv_lateral[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            #Res_Block('leaky', channel),
                                            #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                            #channel_attention(channel),
                                            )   
                                                                                     
                self.unet_fusion[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                                            Res_Block('leaky', channel),
                                            Res_Block('leaky', channel),
                                            )

                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    self.conv_ORB[level] = nn.Sequential(
                                            Res_Block('leaky', channel),
                                            #Res_Block('leaky', channel),
                                            channel_attention(channel),
                                            )

                self.ORB = nn.Sequential(
                                        Res_Block('leaky', channel),
                                        Res_Block('leaky', channel),
                                        )

                self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)


            self.non_linear_transform[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    Res_Block('leaky', channel),
                                                    )
        
        
        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####

        self.feature_list = []
        self.reblur_filter_list = []
        x = encode[f'fpn_feature_lv3']
        decode_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            level = f'fpn_feature_lv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = decode_result
                residual_feature = x

            styled_feature = self.non_linear_transform[deconv](self.conv_up[deconv](x))

            if i == 3:
                decode_result = self.unet_fusion[deconv](styled_feature)

            elif i != 1:
                decode_result = self.unet_fusion[deconv](styled_feature + self.conv_lateral[deconv](encode[level]))
                
            else:
                decode_result = self.unet_fusion[deconv](styled_feature + self.conv_lateral[deconv](encode[level]))

                for j in range(self.num_ORB):
                    deconv_level = f'Deconv{j + 1}'
                    decode_result = self.conv_ORB[deconv_level](decode_result)
                    
                decode_result = self.predict(self.ORB(decode_result))

            self.feature_list.append(decode_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}

        return decode_dict




# 1-2-4
class Patch_Unet_Decofder(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.reblur_attention_lv1 = nn.ModuleDict()
        self.reblur_attention_lv2 = nn.ModuleDict()
        self.reblur_attention_lv3 = nn.ModuleDict()

        self.non_linear_transform = nn.ModuleDict()

        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.unet_fusion = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 5

        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]

        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            ratio = ratio_list[i-1]
            if i == 3:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential(
                                        nn.Conv2d(channel, channel, kernel_size = 3, padding=1))
            else:
                channel = channel_list[i-1]
                self.conv_up[deconv] = nn.Sequential( 
                        nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
                        nn.Conv2d(channel, channel, kernel_size=3,padding=1))

            if i == 3:
                self.unet_fusion[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
                                            Res_Block('leaky', channel//2),
                                            Res_Block('leaky', channel//2),
                                            )

            elif i != 1:
                self.conv_lateral[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            Res_Block('leaky', channel),
                                            #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                            channel_attention(channel),
                                            )  

                self.unet_fusion[deconv] = nn.Sequential(
                                            channel_attention(channel * 2),
                                            nn.Conv2d(channel * 2, channel//2, kernel_size=3, stride=1, padding=1),
                                            Res_Block('leaky', channel//2),
                                            Res_Block('leaky', channel//2),
                                            )
            else:
                self.conv_lateral[deconv] = nn.Sequential(
                                            channel_attention(channel),
                                            Res_Block('leaky', channel),
                                            #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                            channel_attention(channel),
                                            )   
                                                                                     
                self.unet_fusion[deconv] = nn.Sequential(
                                            channel_attention(channel * 2),
                                            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
                                            Res_Block('leaky', channel),
                                            Res_Block('leaky', channel),
                                            )

                for j in range(self.num_ORB):
                    level = f'Deconv{j + 1}'
                    self.conv_ORB[level] = nn.Sequential(
                                            Res_Block('leaky', channel),
                                            #Res_Block('leaky', channel),
                                            channel_attention(channel),
                                            )

                self.ORB = nn.Sequential(
                                        Res_Block('leaky', channel),
                                        Res_Block('leaky', channel),
                                        )

                self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)


            self.non_linear_transform[deconv] = nn.Sequential(
                                                    Res_Block('leaky', channel),
                                                    Res_Block('leaky', channel),
                                                    )
        
        
        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####

        self.feature_list = []
        self.reblur_filter_list = []
        x = encode[f'fpn_feature_lv3']
        decode_result = None
        for i in range(3,0,-1):
            deconv = f'Deconv{i}'
            level = f'fpn_feature_lv{i}'
            if i == 3:
                residual_feature = x
            else:
                x = decode_result
                residual_feature = x

            styled_feature = self.non_linear_transform[deconv](self.conv_up[deconv](x))

            if i == 3:
                decode_result = self.unet_fusion[deconv](styled_feature)

            elif i != 1:
                decode_result = self.unet_fusion[deconv](
                    torch.cat((styled_feature, self.conv_lateral[deconv](encode[level])), dim = 1))
                
            else:
                decode_result = self.unet_fusion[deconv](
                    torch.cat((styled_feature, self.conv_lateral[deconv](encode[level])), dim = 1))

                for j in range(self.num_ORB):
                    deconv_level = f'Deconv{j + 1}'
                    decode_result = self.conv_ORB[deconv_level](decode_result)
                    
                decode_result = self.predict(self.ORB(decode_result))

            self.feature_list.append(decode_result)

        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}

        return decode_dict





# 1-2-4
class Patch_Unet_Decoder(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.reblur_attention_lv1 = nn.ModuleDict()
        self.reblur_attention_lv2 = nn.ModuleDict()
        self.reblur_attention_lv3 = nn.ModuleDict()

        self.non_linear_transform = nn.ModuleDict()

        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.unet_fusion = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 5

        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]
        
        channel = opt.channel * 4

        ## level 3
        self.Decoder_level3_unet = nn.Sequential(
                                channel_attention(channel),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
                                Res_Block('leaky', channel//2),
                                Res_Block('leaky', channel//2),
                                )

        ## level 2
        channel = opt.channel * 2
        self.Decoder_level2_unet = nn.Sequential(
                                channel_attention(channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
                                Res_Block('leaky', channel//2),
                                Res_Block('leaky', channel//2),
                                )

        self.Decoder_level2_lateral = nn.Sequential(
                                    channel_attention(channel),
                                    #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    Res_Block('leaky', channel),
                                    channel_attention(channel),
                                    ) 

        self.Decoder_level2_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                )

        ## level 1
        channel = opt.channel
        self.Decoder_level1_unet = nn.Sequential(
                                channel_attention(channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),                             
                                )

        self.Decoder_level1_lateral = nn.Sequential(
                                    channel_attention(channel),
                                    #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    Res_Block('leaky', channel),
                                    channel_attention(channel),
                                    ) 

        self.Decoder_level1_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                )


        self.Decoder_level1_ORB = nn.Sequential(* ([channel_attention(channel), Res_Block('leaky', channel),]) * self.num_ORB )
        self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.reblur_filter_list = []
        x = encode['fpn_feature_lv3']

        ## level 3
        decode_level3_result = self.Decoder_level3_unet(x)

        ## level 2
        lateral = self.Decoder_level2_lateral(encode['fpn_feature_lv2'])
        up_decode_level3_result = self.Decoder_level2_up(decode_level3_result)
        cat_feteature = torch.cat((up_decode_level3_result, lateral), dim = 1)
        decode_level2_result = self.Decoder_level2_unet(cat_feteature)

        ## level 1
        lateral = self.Decoder_level1_lateral(encode['fpn_feature_lv1'])
        up_decode_level2_result = self.Decoder_level1_up(decode_level2_result)
        cat_feteature = torch.cat((up_decode_level2_result, lateral), dim = 1)
        decode_level1_result = self.Decoder_level1_unet(cat_feteature)

        ## predict
        decode_result = self.predict(self.Decoder_level1_ORB(decode_level1_result))


        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}

        return decode_dict

# 1-2-4
class Patch_Unet_Decoder_1_2_4(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
        self.reblur_attention_lv1 = nn.ModuleDict()
        self.reblur_attention_lv2 = nn.ModuleDict()
        self.reblur_attention_lv3 = nn.ModuleDict()

        self.non_linear_transform = nn.ModuleDict()

        self.attention_ORB = nn.ModuleDict()
        self.conv_ORB = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.predict_ORB = nn.ModuleDict()
        self.attention_predict_ORB = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.unet_fusion = nn.ModuleDict()

        self.motion_channel = 3 * (opt.per_pix_kernel ** 2) * opt.Recurrent_times
        self.num_ORB = 3
        self.kernel = 5

        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]
        
        channel = opt.channel * 4

        ## level 3
        self.Decoder_level3_unet = nn.Sequential(
                                #SP_SFT(opt, channel),
                                nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel//2, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Res_Block('leaky', channel//2),
                                Res_Block('leaky', channel//2),
                                )

        ## level 2
        channel = opt.channel * 2
        self.Decoder_level2_unet = nn.Sequential(
                                #SP_SFT(opt, channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel//2, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Res_Block('leaky', channel//2),
                                Res_Block('leaky', channel//2),
                                )

        self.Decoder_level2_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 

        self.Decoder_level2_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                )

        ## level 1
        channel = opt.channel
        self.Decoder_level1_unet = nn.Sequential(
                                #channel_attention(channel*2),
                                #SP_SFT(opt, channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                Res_Block('leaky', channel),
                                Res_Block('leaky', channel),                             
                                )

        self.Decoder_level1_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    #nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 

        self.Decoder_level1_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                Res_Block('leaky', channel),
                                )


        self.Decoder_level1_ORB = nn.Sequential(* ([Res_Block('leaky', channel),]) * self.num_ORB )
        self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        self.reblur_filter_list = []
        x = encode['fpn_feature_lv3']

        ## level 3
        decode_level3_result = self.Decoder_level3_unet(x)

        ## level 2
        lateral = self.Decoder_level2_lateral(encode['fpn_feature_lv2'])
        up_decode_level3_result = self.Decoder_level2_up(decode_level3_result)
        cat_feteature = torch.cat((up_decode_level3_result, lateral), dim = 1)
        decode_level2_result = self.Decoder_level2_unet(cat_feteature)

        ## level 1
        lateral = self.Decoder_level1_lateral(encode['fpn_feature_lv1'])
        up_decode_level2_result = self.Decoder_level1_up(decode_level2_result)
        cat_feteature = torch.cat((up_decode_level2_result, lateral), dim = 1)
        decode_level1_result = self.Decoder_level1_unet(cat_feteature)

        ## predict
        decode_result = self.predict(self.Decoder_level1_ORB(decode_level1_result))


        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}

        return decode_dict


# 1-2-4
class Patch_Unet_Decoder_1_2_4_8f(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
    
        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]
        self.kernel = 5
        self.num_ORB = 5

        ## level 4
        channel = opt.channel * 4
        self.Decoder_level4_unet = nn.Sequential(
                                #channel_attention(channel),
                                nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                )

        ## level 3
        channel = opt.channel * 4
        self.Decoder_level3_unet = nn.Sequential(
                                #channel_attention(channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel//2, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel//2),
                                Res_Block('leaky', channel//2),
                                )

        self.Decoder_level3_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    #Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 
        #self.Decoder_level3_predict = SAM(channel//2)

        ## level 2
        channel = opt.channel * 2
        self.Decoder_level2_unet = nn.Sequential(
                                #channel_attention(channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel//2, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel//2),
                                Res_Block('leaky', channel//2),
                                )

        self.Decoder_level2_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    #Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 

        self.Decoder_level2_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                #Res_Block('leaky', channel),
                                )
        #self.Decoder_level2_predict = SAM(channel//2)

        ## level 1
        channel = opt.channel
        self.Decoder_level1_unet = nn.Sequential(
                                #channel_attention(channel*2),
                                nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel),
                                Res_Block('leaky', channel),
                                nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                #Res_Block('leaky', channel),
                                Res_Block('leaky', channel),                             
                                )

        self.Decoder_level1_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    #Res_Block('leaky', channel),
                                    channel_attention(channel),
                                    ) 

        self.Decoder_level1_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                #Res_Block('leaky', channel),
                                )
        #self.Decoder_level1_predict = SAM(channel)


        self.Decoder_level1_ORB = nn.Sequential(* ([ Res_Block('leaky', channel),]) * self.num_ORB )
        self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        #multi_scale_input = [F.interpolate(input, scale_factor=0.25), F.interpolate(input, scale_factor=0.5), input]
        self.reblur_filter_list = []
        self.decode_predict = []
        x = encode['fpn_feature_lv4']

        ## level 4
        decode_level4_result = self.Decoder_level4_unet(x)

        ## level 3
        lateral = self.Decoder_level3_lateral(encode['fpn_feature_lv3'])
        cat_feteature = torch.cat((decode_level4_result, lateral), dim = 1)
        decode_level3_result = self.Decoder_level3_unet(cat_feteature)
        #decode_level3_result = self.Decoder_level3_predict(decode_level3_result, F.interpolate(input, scale_factor=0.25))
        #self.decode_predict.append(decode_level3_result['decode_predict'])
        #decode_level3_result = decode_level3_result['attention_feature']

        ## level 2
        lateral = self.Decoder_level2_lateral(encode['fpn_feature_lv2'])
        up_decode_level3_result = self.Decoder_level2_up(decode_level3_result)
        cat_feteature = torch.cat((up_decode_level3_result, lateral), dim = 1)
        decode_level2_result = self.Decoder_level2_unet(cat_feteature)
        #decode_level2_result = self.Decoder_level2_predict(decode_level2_result, F.interpolate(input, scale_factor=0.5))
        #self.decode_predict.append(decode_level2_result['decode_predict'])
        #decode_level2_result = decode_level2_result['attention_feature']

        ## level 1
        lateral = self.Decoder_level1_lateral(encode['fpn_feature_lv1'])
        up_decode_level2_result = self.Decoder_level1_up(decode_level2_result)
        cat_feteature = torch.cat((up_decode_level2_result, lateral), dim = 1)
        decode_level1_result = self.Decoder_level1_unet(cat_feteature)
        #decode_level1_result = self.Decoder_level1_predict(decode_level1_result, input)
        #self.decode_predict.append(decode_level1_result['decode_predict'])
        #decode_level1_result = decode_level1_result['attention_feature']

        ## predict
        decode_result = self.predict(self.Decoder_level1_ORB(decode_level1_result))


        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}#, 'decode_predict' : self.decode_predict}

        return decode_dict
    

# 1-2-4
class Patch_Unet_Decoder_1_2_4_8(nn.Module):
    #####
        # Let motion 'localize' affect the feature, while doing non-linear transform
    #####
    def __init__(self, opt):
        super().__init__()      
    
        channel_list = [opt.channel, opt.channel*2, opt.channel*4]
        ratio_list = [1, 2, 4]
        self.kernel = 5
        self.num_ORB = 2

        ## level 4
        channel = opt.channel * 4
        self.Decoder_level4_unet = nn.Sequential(
                                        #nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        )

        ## level 3
        channel = opt.channel * 4
        self.Decoder_level3_unet = nn.Sequential(
                                        nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel//2, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        )

        self.Decoder_level3_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    #Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 
        #self.Decoder_level3_predict = SAM(channel//2)

        ## level 2
        channel = opt.channel * 2
        self.Decoder_level2_unet = nn.Sequential(
                                        nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel//2, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        )

        self.Decoder_level2_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    #Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 

        self.Decoder_level2_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                #Res_Block('leaky', channel),
                                )
        #self.Decoder_level2_predict = SAM(channel//2)

        ## level 1
        channel = opt.channel
        self.Decoder_level1_unet = nn.Sequential(
                                        nn.Conv2d(channel*2, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        Component_Block(opt, in_c = channel, out_c = channel, stride_len = 1),
                                        nn.Conv2d(channel, channel, kernel_size=(self.kernel,self.kernel), stride=1, padding=self.kernel//2),
                                        )

        self.Decoder_level1_lateral = nn.Sequential(
                                    #channel_attention(channel),
                                    nn.Conv2d(channel, channel, kernel_size=1, stride=1),
                                    #Res_Block('leaky', channel),
                                    #channel_attention(channel),
                                    ) 

        self.Decoder_level1_up = nn.Sequential(
                                nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                nn.Conv2d(channel, channel, kernel_size = 3, padding=1),
                                #Res_Block('leaky', channel),
                                )
        #self.Decoder_level1_predict = SAM(channel)


        self.Decoder_level1_ORB = nn.Sequential(* ([ Res_Block('leaky', channel),]) * self.num_ORB )
        self.predict = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)

        
    def forward(self, encode):
        ####
        # input encoder_feature (x) size : (batch, channel * 4, H/4, W/4)
        # input ef_lv2 size : (batch, channel * 2, H/2, W/2)
        # input ef_lv1 size : (batch, channel, H, W)
        # motion feature size : (batch, channel * 4, H/4, W/4)
        ####
        #multi_scale_input = [F.interpolate(input, scale_factor=0.25), F.interpolate(input, scale_factor=0.5), input]
        self.reblur_filter_list = []
        self.decode_predict = []
        x = encode['fpn_feature_lv4']

        ## level 4
        decode_level4_result = self.Decoder_level4_unet(x)

        ## level 3
        lateral = self.Decoder_level3_lateral(encode['fpn_feature_lv3'])
        cat_feteature = torch.cat((decode_level4_result, lateral), dim = 1)
        decode_level3_result = self.Decoder_level3_unet(cat_feteature)

        ## level 2
        lateral = self.Decoder_level2_lateral(encode['fpn_feature_lv2'])
        up_decode_level3_result = self.Decoder_level2_up(decode_level3_result)
        cat_feteature = torch.cat((up_decode_level3_result, lateral), dim = 1)
        decode_level2_result = self.Decoder_level2_unet(cat_feteature)

        ## level 1
        lateral = self.Decoder_level1_lateral(encode['fpn_feature_lv1'])
        up_decode_level2_result = self.Decoder_level1_up(decode_level2_result)
        cat_feteature = torch.cat((up_decode_level2_result, lateral), dim = 1)
        decode_level1_result = self.Decoder_level1_unet(cat_feteature)

        ## predict
        decode_result = self.predict(self.Decoder_level1_ORB(decode_level1_result))


        # Deconv3
        # input size : (batch, channel * 4, H/4, W/4)

        # Deconv2
        # input size : (batch, channel * 2, H/2, W/2)

        # Deconv1
        # input size : (batch, channel * 1, H, W)  

        decode_dict = {'deblur' : decode_result, 'reblur_filter' : self.reblur_filter_list}#, 'decode_predict' : self.decode_predict}

        return decode_dict