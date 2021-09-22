import sub_modules

# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and not classname.find('Conv1d') != -1 and not classname.find('ConvLSTM') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size()).cuda()


def define_G(opt, isGAN):
    pass

def define_deblur_net(opt):
    use_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.G == 'Patch_Deblur_FPN_VAE':
        netG = Patch_Deblur_FPN_VAE(opt)

    elif opt.G == 'BRRMv4ALL_DMPHN_1_2_4_8':
        netG = BRRMv4ALL_DMPHN_1_2_4_8(opt)

    return netG.apply(weights_init).to(use_gpu)


##############################################################################
# Classes
##############################################################################

class BRRM_framework(nn.Module):
    '''
        1. add convlstm to add time-series property
        2. per-pixel kernel = 3
        3. recurrent time = 3
    '''
    def __init__(self, opt):
        super().__init__()
        self.deblur_net = define_deblur_net(opt)
        self.reblur_module = Reblurring_Module(opt)
        self.recurrent_times = opt.Recurrent_times
        self.convlstm = nn.ModuleDict()
        for i in range(self.recurrent_times-1):
            level = f'level_{i+1}'
            self.convlstm[level] = ConvLSTM(input_dim = opt.per_pix_kernel **2, hidden_dim=opt.per_pix_kernel **2, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=True)

    def generate_motion(self, motion_feature):
            '''
                use self.recurrent_time - 1 times of self.ConvLSTM
                return :
                    self.reblur_filter : [motion feature in timestamp 1, motion feature in timestamp 2, .., motion feature in timestamp 5]
            '''
            hidden_state = None
            self.reblur_filter = []
            for j in range(len(motion_feature)):
                motionInfo = motion_feature[j]
                self.reblur_filter.append(motionInfo)
                for i in range(self.recurrent_times - 1):
                    level = f'level_{i+1}'
                    # repeat generate the motionInfo
                    motionInfo, hidden_state = self.convlstm[level](torch.unsqueeze(motionInfo, 1), hidden_state)
                    self.reblur_filter.append(torch.squeeze(motionInfo[0], 1))
                    motionInfo = torch.squeeze(motionInfo[0], 1)
    
    def forward(self, input, sharp):
        '''
            return 
                1. reblur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
                2. deblur : deblur result
        '''
        deblur_dict = self.deblur_net(input)
        sharp_reblur, deblur_reblur = None, None
        self.generate_motion(deblur_dict['reblur_filter'])

        sharp_reblur = self.reblur_module(sharp, self.reblur_filter)
        deblur_reblur = self.reblur_module(deblur_dict['deblur'], self.reblur_filter)

        return {'deblur': [deblur_dict['deblur']], 'reblur' : {'sharp_reblur' : sharp_reblur, 'deblur_reblur':deblur_reblur}}

if __name__ == '__main__':
    print('hello')