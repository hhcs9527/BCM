import sub_modules

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


class BR_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.BCM = BCM_block(opt, opt.channel)
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
        x = self.BCM(x)
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

class BRR_Encoder(nn.Module):
    def __init__(self, opt, level):
        super().__init__()
        self.level = level
        #Conv1
        self.layer1 = nn.Conv2d(3, opt.channel, kernel_size=3, padding=1)
        ###
        self.BCM = BCM_block(opt, opt.channel)
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
        x = self.BCM(x)
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

        residual_attention_featurex1 = self.residual_attention_featurex1(attention_feature)
        residual_attention_featurex2 = self.residual_attention_featurex2(attention_feature)
        residual_attention_featurex4 = self.residual_attention_featurex4(attention_feature)

        return {'encode_feature' : x, 'reblur_filter' : self.reblur_filter,
        'residual_attention_featurex1' : residual_attention_featurex1, 'residual_attention_featurex2' : residual_attention_featurex2, 'residual_attention_featurex4' : residual_attention_featurex4,
        f'lv{self.level}_low': conv, f'lv{self.level}_mid': blur, f'lv{self.level}_high': sharpen}