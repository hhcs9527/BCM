import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sub_modules.attention_zoo import *
from sub_modules.component_block import *

class BiFPN_layer(nn.Module):
    """
        Reference : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/15403b5371a64defb2a7c74e162c6e880a7f462c/efficientdet/model.py#L55
    """
    def __init__(self, opt, out_channel_list, stride_list):
        super().__init__()
        self.lateral_conv = nn.ModuleDict()
        self.lateral_intermid_conv = nn.ModuleDict()
        self.Residual = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_down = nn.ModuleDict()
        self.conv_pred = nn.ModuleDict()
        self.prediction_weight = {}
        self.conv_predict = nn.ModuleDict()
        self.intermid_weight = {}
        self.FPN_level = 6
        self.epsilon = 1e-4
        self.leaky = nn.LeakyReLU(0.2, True)
        self.motion_channel = 3 * (opt.per_pix_kernel **2 ) * opt.Recurrent_times
        self.sp_attention = nn.ModuleDict()
        self.num_sp_attention = 2
        self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'

        for i in range(2, self.FPN_level + 1):
            level = f'level_{i}'
            self.conv_pred[level] = Residual_module(nn.Sequential(Base_Res_Block(opt, out_channel_list[i-1]), Base_Res_Block(opt, out_channel_list[i-1]),))
            if i in [2]:
                att_mod_list = nn.ModuleDict()
                for j in range(self.num_sp_attention):
                    att = f'att_{j+1}'
                    att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, 2)
        
                self.sp_attention[level] = att_mod_list

            elif i in [3, 4, 5, 6]:
                att_mod_list = nn.ModuleDict()
                for j in range(self.num_sp_attention):
                    att = f'att_{j+1}'
                    att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, 4)
        
                self.sp_attention[level] = att_mod_list

            self.conv_predict[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, padding = 1, stride = 1)
            if i in [3, 4, 5]:
                self.intermid_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)
                self.prediction_weight[level] = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True).to(self.isgpu)
                self.lateral_conv[level] = Base_Res_Block(opt, out_channel_list[i-1]) #nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, padding = 1, dilation = 1)
                self.Residual[level] = Base_Res_Block(opt, out_channel_list[i-1])
                self.lateral_intermid_conv[level] = Base_Res_Block(opt, out_channel_list[i-1])

            else:
                self.prediction_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)

            if i != 2:
                if i == 3:
                    self.conv_down[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                            nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            self.leaky)         
                else:
                    self.conv_down[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            self.leaky)  
            if i != 6:
                s = 1
                if i == 2:
                    s = 2
                self.conv_up[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i], kernel_size=3, padding = 1, stride = s),
                                      self.leaky)     


    def forward(self, FPN_info, motion):
        """
        intermid_weight means : Pk_1
        prediction_weight means : Pk_2

        illustration of a minimal bifpn unit

        sp_attention -> P6_0 -------------------------> P6_2 -> sp_attention
                         |-------------|                ↑
                                       ↓                |
        sp_attention -> P5_0 ---------> P5_1 ---------> P5_2 -> sp_attention
                         |-------------|--------------↑ ↑
                                       ↓                |
        sp_attention -> P4_0 ---------> P4_1 ---------> P4_2 -> sp_attention
                         |-------------|--------------↑ ↑
                                       ↓                |
        sp_attention -> P3_0 ---------> P3_1 ---------> P3_2 -> sp_attention
                         |-------------|--------------↑ ↑
                                       |--------------↓ |
        sp_attention -> P2_0 -------------------------> P2_2 -> sp_attention

        """        


        intermid_info = {}
        for i in range(self.FPN_level-1, 2, -1):
            level = f'level_{i}'
            upper_level = f'level_{i+1}'
            fpn_feature = f'fpn_feature_lv{i}'
            upper_fpn_feature = f'fpn_feature_lv{i+1}'
            if i == self.FPN_level-1:
                from_lateral = self.lateral_conv[level](FPN_info[fpn_feature])
                from_top = self.conv_down[upper_level](FPN_info[upper_fpn_feature])

                w = (self.intermid_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                intermid_info[f'P{self.FPN_level-1}_1'] = norm_w[0] * from_lateral + norm_w[1] * from_top
                intermid_info[f'P{self.FPN_level-1}_2'] = self.Residual[level](intermid_info[f'P{self.FPN_level-1}_1']) 
                   
            else:
                from_lateral = self.lateral_conv[level](FPN_info[fpn_feature])
                from_top = self.conv_down[upper_level](intermid_info[f'P{i+1}_1']) 

                w = (self.intermid_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                intermid_info[f'P{i}_1'] = norm_w[0] * from_lateral + norm_w[1] * from_top
                intermid_info[f'P{i}_2'] = self.Residual[level](intermid_info[f'P{i}_1']) 


        for i in range(self.FPN_level-1, 2, -1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            for j in range(self.num_sp_attention):
                att = f'att_{j+1}'
                FPN_info[fpn_feature] = self.sp_attention[level][att](FPN_info[fpn_feature], motion)

        
        predict_FPN_info = {}
        for i in range(2, self.FPN_level + 1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'  

            if i == 2:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = FPN_info[fpn_feature]
                from_top = self.conv_down['level_3'](intermid_info['P3_1'])
                predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_lateral + norm_w[1] * from_top + FPN_info[fpn_feature])
                #predict_FPN_info[fpn_feature] = from_lateral + from_top + FPN_info[fpn_feature]
            
            elif i == self.FPN_level:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = FPN_info[fpn_feature]
                from_down = self.conv_up[f'level_{self.FPN_level-1}'](predict_FPN_info[last_fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_lateral + norm_w[1] * from_down + FPN_info[fpn_feature])
                #predict_FPN_info[fpn_feature] = from_lateral + from_down + FPN_info[fpn_feature]
            
            else:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = FPN_info[fpn_feature]
                from_down = self.conv_up[f'level_{i-1}'](predict_FPN_info[last_fpn_feature])
                from_intermid = self.lateral_intermid_conv[level](intermid_info[f'P{i}_2'])
                predict_FPN_info[fpn_feature] = self.conv_pred[level](norm_w[0] * from_lateral + norm_w[1] * from_down + norm_w[2] * from_intermid + FPN_info[fpn_feature])
                #predict_FPN_info[fpn_feature] = from_lateral + from_down + from_intermid + FPN_info[fpn_feature]

        return predict_FPN_info


# for three layers
class BiFPN(nn.Module):
    """
        Reference : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/15403b5371a64defb2a7c74e162c6e880a7f462c/efficientdet/model.py#L55
    """
    def __init__(self, opt, out_channel_list, stride_list):
        super().__init__()
        self.lateral_conv = nn.ModuleDict()
        self.Residual = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_down = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_lateral_intermid = nn.ModuleDict()
        self.conv_lateral_predict = nn.ModuleDict()
        self.prediction_weight = {}
        self.conv_predict = nn.ModuleDict()
        self.intermid_weight = {}
        self.FPN_level = 4
        self.epsilon = 1e-8
        self.leaky = nn.LeakyReLU(0.2, True)
        self.motion_channel = 3 * (opt.per_pix_kernel **2 ) * opt.Recurrent_times
        self.sp_attention_head = nn.ModuleDict()
        self.num_sp_attention = 1
        self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        in_channel_list = [3, opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [1 ,2 ,4 ,4]
        stride_list = [1, 2, 2, 1] # 256, 128, 64, 64

        for i in range(2, self.FPN_level + 1):
            level = f'level_{i}'
            ratio = ratio_list[i-1]

            att_mod_list = nn.ModuleDict()
            for j in range(self.num_sp_attention):
                att = f'att_{j+1}'
                att_mod_list[att] = SP_motion_attention(opt, out_channel_list[i-1], self.motion_channel, ratio)
        
            self.sp_attention_head[level] = att_mod_list

            self.conv_predict[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, padding = 1, stride = 1)
            self.conv_lateral[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, padding = 1, stride = 1)

            if i != 2 and i != self.FPN_level:
                self.intermid_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)
                self.prediction_weight[level] = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True).to(self.isgpu)
                self.conv_lateral_intermid[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, padding = 1, dilation = 1)
                self.conv_lateral_predict[level] = nn.Conv2d(out_channel_list[i-1], out_channel_list[i-1], kernel_size=3, padding = 1, dilation = 1)
            else:
                self.prediction_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)

            if i != 2:
                if i == 3:
                    self.conv_down[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                            nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            self.leaky)         
                else:
                    self.conv_down[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i-2], kernel_size=3, padding = 1, dilation = 1),
                                            self.leaky)  
    
            if i == 2:
                self.conv_up[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i], kernel_size=3, padding = 1, stride = 2),
                                            self.leaky)     
            elif i != self.FPN_level:
                self.conv_up[level] = nn.Sequential(nn.Conv2d(out_channel_list[i-1], out_channel_list[i], kernel_size=3, padding = 1, stride = 1),
                                            self.leaky)   

    def forward(self, FPN_info, motion):
        """
        intermid_weight means : Pk_1
        prediction_weight means : Pk_2

        illustration of a minimal bifpn unit

        sp_attention -> P4_0 ------------------------> P4_2 -> sp_attention
                         |-------------|                ↑
                                       ↓                |
        sp_attention -> P3_0 ---------> P3_1 ---------> P3_2 -> sp_attention
                         |-------------|--------------↑ ↑
                                       |--------------↓ |
        sp_attention -> P2_0 -------------------------> P2_2 -> sp_attention

        """        

        for i in range(2, self.FPN_level + 1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            for j in range(self.num_sp_attention):
                att = f'att_{j+1}'
                FPN_info[fpn_feature] = self.sp_attention_head[level][att](FPN_info[fpn_feature], motion)


        intermid_info = {}
        for i in range(self.FPN_level-1, 2, -1):
            level = f'level_{i}'
            upper_level = f'level_{i+1}'
            fpn_feature = f'fpn_feature_lv{i}'
            upper_fpn_feature = f'fpn_feature_lv{i+1}'
            if i == self.FPN_level-1:
                from_lateral = self.conv_lateral[level](FPN_info[fpn_feature])
                from_top = self.conv_down[upper_level](FPN_info[upper_fpn_feature])

                #w = (self.intermid_weight[level])
                #norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                #intermid_info[f'P{i}_1'] = norm_w[0] * from_lateral + norm_w[1] * from_top
                intermid_info[f'P{i}_1'] = from_lateral + from_top
                intermid_info[f'P{i}_2'] = self.conv_lateral_intermid[level](intermid_info[f'P{i}_1']) 
                   
            else:
                from_lateral = self.conv_lateral[level](FPN_info[fpn_feature])
                from_top = self.conv_down[upper_level](intermid_info[f'P{i+1}_1']) 

                #w = (self.intermid_weight[level])
                #norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                #intermid_info[f'P{i}_1'] = norm_w[0] * from_lateral + norm_w[1] * from_top
                intermid_info[f'P{i}_1'] = from_lateral + from_top
                intermid_info[f'P{i}_2'] = self.conv_lateral_intermid[level](intermid_info[f'P{i}_1']) 


        predict_FPN_info = {}
        for i in range(2, self.FPN_level + 1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'  

            if i == 2:
                #w = (self.prediction_weight[level])
                #norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = self.conv_lateral[level](FPN_info[fpn_feature])
                from_top = self.conv_down[f'level_{i+1}'](intermid_info[f'P{i+1}_2'])
                #predict_FPN_info[fpn_feature] = self.conv_predict[level](norm_w[0] * from_lateral + norm_w[1] * from_top + FPN_info[fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_predict[level](from_lateral + from_top)
            
            elif i == self.FPN_level:
                #w = (self.prediction_weight[level])
                #norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = self.conv_lateral[level](FPN_info[fpn_feature])
                from_bottom = self.conv_up[f'level_{i-1}'](predict_FPN_info[last_fpn_feature])
                #predict_FPN_info[fpn_feature] = self.conv_predict[level](norm_w[0] * from_lateral + norm_w[1] * from_bottom + FPN_info[fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_predict[level](from_lateral + from_bottom)

            else:
                #w = (self.prediction_weight[level])
                #norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = self.conv_lateral_predict[level](FPN_info[fpn_feature])
                from_down = self.conv_up[f'level_{i-1}'](predict_FPN_info[last_fpn_feature])
                from_intermid = intermid_info[f'P{i}_2']
                #predict_FPN_info[fpn_feature] = self.conv_predict[level](norm_w[0] * from_lateral + norm_w[1] * from_down + norm_w[2] * from_intermid + FPN_info[fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_predict[level](from_lateral + from_down + from_intermid)

        return predict_FPN_info

# for three layers
class bifpn_multi_scale_fusion(nn.Module):
    """
        Reference : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/15403b5371a64defb2a7c74e162c6e880a7f462c/efficientdet/model.py#L55
    """
    def __init__(self, opt, out_channel_list, stride_list):
        super().__init__()
        self.Residual = nn.ModuleDict()
        self.conv_up = nn.ModuleDict()
        self.conv_down = nn.ModuleDict()
        self.conv_lateral = nn.ModuleDict()
        self.conv_lateral_intermid = nn.ModuleDict()
        self.prediction_weight = {}
        self.conv_predict = nn.ModuleDict()
        self.intermid_weight = {}
        self.num_scale = 3
        self.scale_list = [2 ** i for i in range(self.num_scale)]
        self.epsilon = 1e-8
        self.leaky = nn.LeakyReLU(0.2, True)
        self.motion_channel = 3 * (opt.per_pix_kernel **2 ) * opt.Recurrent_times
        self.sp_attention_head = nn.ModuleDict()
        self.num_sp_attention = 1
        self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'

        out_channel_list = [opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        in_channel_list = [3, opt.channel, opt.channel*2, opt.channel*4, opt.channel*4]
        ratio_list = [2 ** i for i in range(self.num_scale)]#[1 ,2 ,4 ,4]
        stride_list = [1, 2, 2, 1] # 256, 128, 64, 64
        channel = opt.channel 

        for i in range(1, self.num_scale + 1):
            level = f'level_{i}'
            ratio = ratio_list[i-1]

            att_mod_list = nn.ModuleDict()
            for j in range(self.num_sp_attention):
                att = f'att_{j+1}'
                att_mod_list[att] = SP_motion_attention(opt,channel, self.motion_channel, ratio)
        
            self.sp_attention_head[level] = att_mod_list

            self.conv_predict[level] = nn.Conv2d(channel, channel, kernel_size=3, padding = 1, stride = 1)
            self.conv_lateral[level] = nn.Conv2d(channel, channel, kernel_size=3, padding = 1, stride = 1)

            if i != 1 and i != self.num_scale:
                self.intermid_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)
                self.prediction_weight[level] = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True).to(self.isgpu)
                self.conv_lateral_intermid[level] = nn.Conv2d(channel, channel, kernel_size=3, padding = 1, dilation = 1)
                #self.Residual[level] = Base_Res_Block(opt, channel)

            else:
                self.prediction_weight[level] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True).to(self.isgpu)

            
            if i != 1:
                self.conv_down[level] = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(channel, channel, kernel_size=3, padding = 1, dilation = 1),
                                        self.leaky)         

            if i != self.num_scale:
                self.conv_up[level] = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding = 1, stride = 2),
                                        nn.Conv2d(channel, channel, kernel_size=3, padding = 1, dilation = 1),
                                        self.leaky)     

    def forward(self, FPN_info, motion):
        """
        intermid_weight means : Pk_1
        prediction_weight means : Pk_2

        illustration of a minimal bifpn unit

        sp_attention -> P4_0 ------------------------> P4_2 -> sp_attention
                         |-------------|                ↑
                                       ↓                |
        sp_attention -> P3_0 ---------> P3_1 ---------> P3_2 -> sp_attention
                         |-------------|--------------↑ ↑
                                       |--------------↓ |
        sp_attention -> P2_0 -------------------------> P2_2 -> sp_attention

        """        

        for i in range(1, self.num_scale):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            for j in range(self.num_sp_attention):
                att = f'att_{j+1}'
                FPN_info[fpn_feature] = self.sp_attention_head[level][att](FPN_info[fpn_feature], motion)
                
        intermid_info = {}
        for i in range(self.num_scale - 1, 1, -1):
            level = f'level_{i}'
            upper_level = f'level_{i+1}'
            fpn_feature = f'fpn_feature_lv{i}'
            top_fpn_feature = f'fpn_feature_lv{i+1}'
            if i == self.num_scale - 1:
                from_lateral = self.conv_lateral_intermid[level](FPN_info[fpn_feature])
                from_top = self.conv_down[upper_level](FPN_info[top_fpn_feature])

                w = (self.intermid_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                intermid_info[f'P{i}_1'] = norm_w[0] * from_lateral + norm_w[1] * from_top
                intermid_info[f'P{i}_2'] = self.conv_lateral[level](intermid_info[f'P{i}_1']) 
                   
            else:
                from_lateral = self.conv_lateral_intermid[level](FPN_info[fpn_feature])
                from_top = self.conv_down[upper_level](intermid_info[f'P{i+1}_1']) 

                w = (self.intermid_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)
                intermid_info[f'P{i}_1'] = norm_w[0] * from_lateral + norm_w[1] * from_top
                intermid_info[f'P{i}_2'] = self.conv_lateral[level](intermid_info[f'P{i}_1']) 


        predict_FPN_info = {}
        for i in range(1, self.num_scale + 1):
            level = f'level_{i}'
            fpn_feature = f'fpn_feature_lv{i}'
            last_fpn_feature = f'fpn_feature_lv{i-1}'  

            if i == 1:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = FPN_info[fpn_feature]
                from_top = self.conv_down[f'level_{i+1}'](intermid_info[f'P{i+1}_2'])
                predict_FPN_info[fpn_feature] = self.conv_predict[level](norm_w[0] * from_lateral + norm_w[1] * from_top + FPN_info[fpn_feature])
            
            elif i == self.num_scale:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = FPN_info[fpn_feature]
                from_down = self.conv_up[f'level_{i-1}'](predict_FPN_info[last_fpn_feature])
                predict_FPN_info[fpn_feature] = self.conv_predict[level](norm_w[0] * from_lateral + norm_w[1] * from_down + FPN_info[fpn_feature])
            
            else:
                w = (self.prediction_weight[level])
                norm_w = w / (torch.sum(w, dim = 0) + self.epsilon)

                from_lateral = FPN_info[fpn_feature]
                from_down = self.conv_up[f'level_{i-1}'](predict_FPN_info[last_fpn_feature])
                from_intermid = intermid_info[f'P{i}_2']
                predict_FPN_info[fpn_feature] = self.conv_predict[level](norm_w[0] * from_lateral + norm_w[1] * from_down + norm_w[2] * from_intermid + FPN_info[fpn_feature])

        return predict_FPN_info