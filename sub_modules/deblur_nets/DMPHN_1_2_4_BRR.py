import sub_modules

class DMPHN_1_2_4_BRR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        compression_ratio = 1
        self.encoder_lv3 = Encoder(opt) 
        self.encoder_lv2 = Encoder(opt) 
        self.encoder_lv1 = BRR_Encoder(opt, level = 1) 

        self.decoder_lv3 = Decoder(opt)
        self.decoder_lv2 = Decoder(opt)   
        self.decoder_lv1 = BRR_Decoder(opt)
        self.num_split = opt.per_pix_kernel ** 2

    def forward(self, input):
        H = input.size(2)          
        W = input.size(3)
        self.reblur_filters = []
        feature_lv1_dict = {}
        feature_lv2_dict = {}
        feature_lv3_top_dict = {}
        feature_lv3_bot_dict = {}
        key_words = ['lv1_low', 'lv2_low', 'lv3_low', 'lv1_mid', 'lv2_mid', 'lv3_mid', 'lv1_high', 'lv2_high', 'lv3_high']
        self.feature_dict = {}
        self.level_dict = {}
         
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
        for key in feature_lv3_1.keys():
            if key == 'encode_feature':
                feature_lv3_top_dict[key] = torch.cat((feature_lv3_1[key], feature_lv3_2[key]), 3)
                feature_lv3_bot_dict[key] = torch.cat((feature_lv3_3[key], feature_lv3_4[key]), 3)
                # Residual Path like original paper
                feature_lv3 = torch.cat((feature_lv3_top_dict[key], feature_lv3_bot_dict[key]), 2)

            elif key == 'reblur_filter':
                feature_lv3_1['reblur_filter'] = torch.cat((feature_lv3_1['reblur_filter']), dim = 1)
                feature_lv3_2['reblur_filter'] = torch.cat((feature_lv3_2['reblur_filter']), dim = 1)
                feature_lv3_3['reblur_filter'] = torch.cat((feature_lv3_3['reblur_filter']), dim = 1)
                feature_lv3_4['reblur_filter'] = torch.cat((feature_lv3_4['reblur_filter']), dim = 1)
                
                feature_lv3_reblur_filter_top = torch.cat((feature_lv3_1['reblur_filter'], feature_lv3_2['reblur_filter']), 3)
                feature_lv3_reblur_filter_bot = torch.cat((feature_lv3_3['reblur_filter'], feature_lv3_4['reblur_filter']), 3)
                feature_lv3_reblur_filter = torch.cat((feature_lv3_reblur_filter_top, feature_lv3_reblur_filter_bot), 2) 
                split = list(torch.split(feature_lv3_reblur_filter, self.num_split, dim = 1)) 
                self.reblur_filters += split       

            else:
                feature_lv3_top_dict[key] = torch.cat((feature_lv3_1[key], feature_lv3_2[key]), 3)
                feature_lv3_bot_dict[key] = torch.cat((feature_lv3_3[key], feature_lv3_4[key]), 3)

        for key in feature_lv3_1.keys():    
            if key in key_words:
                self.feature_dict[key] = torch.cat((feature_lv3_top_dict[key], feature_lv3_bot_dict[key]), 2)


        residual_lv3_top = self.decoder_lv3(feature_lv3_top_dict)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot_dict)
        self.level_dict['lv3'] = torch.cat((residual_lv3_top, residual_lv3_bot), 2)

        ## level 2
        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        for key in feature_lv2_1.keys():
            if key == 'encode_feature':
                feature_lv2_dict[key] = torch.cat((feature_lv2_1[key], feature_lv2_2[key]), 2) + feature_lv3

            elif key == 'reblur_filter':
                feature_lv2_1['reblur_filter'] = torch.cat((feature_lv2_1['reblur_filter']), dim = 1) 
                feature_lv2_2['reblur_filter'] = torch.cat((feature_lv2_2['reblur_filter']), dim = 1)      
                feature_lv2_reblur_filter = torch.cat((feature_lv2_1['reblur_filter'], feature_lv2_2['reblur_filter']), 2)    
                split = list(torch.split(feature_lv2_reblur_filter, self.num_split, dim = 1))    
                self.reblur_filters += split       

            else:
                feature_lv2_dict[key] = torch.cat((feature_lv2_1[key], feature_lv2_2[key]), 2)

        for key in feature_lv2_1.keys():    
            if key in key_words:
                self.feature_dict[key] = feature_lv2_dict[key]

        residual_lv2 = self.decoder_lv2(feature_lv2_dict)
        self.level_dict['lv2'] = residual_lv2

        ## level 1
        feature_lv1_1 = self.encoder_lv1(images_lv1 + residual_lv2)

        for key in feature_lv1_1.keys():
            if key == 'encode_feature':

                feature_lv1_dict[key] = feature_lv1_1[key] + feature_lv2_dict['encode_feature']

            elif key == 'reblur_filter':
                feature_lv1_1['reblur_filter'] = torch.cat((feature_lv1_1['reblur_filter']), dim = 1)
                split = list(torch.split(feature_lv1_1['reblur_filter'], self.num_split, dim = 1))
                self.reblur_filters += split 

            else:
                feature_lv1_dict[key] = feature_lv1_1[key]

        for key in feature_lv1_1.keys():    
            if key in key_words:
                self.feature_dict[key] = feature_lv1_dict[key]

        deblur_image = self.decoder_lv1(feature_lv1_dict)   
        self.level_dict['lv1'] = deblur_image

        return {'deblur' : deblur_image, 'reblur_filter' : self.reblur_filters}