
    def visualize_motion_feature_map(self, epoch):
        visual_image = transforms.ToTensor()(Image.open('/home/hh/Desktop/GoPro/train/GOPR0384_11_03/blur/002101.png').convert('RGB')).unsqueeze(0).to(self.device)
        x = visual_image
        self.args.isTrain = False
        with torch.no_grad():     
            self.MPGAN.set_pic_input(x)               
            self.MPGAN.test()

            for i in range(len(self.MPGAN.netG.gf.feature_list)):
                _, C, H, W = self.MPGAN.netG.gf.feature_list[i].size()
                map_all = (self.MPGAN.netG.gf.feature_list[i])

                for j in range(C):
                    img = map_all[0][i].unsqueeze(0)
                    self.tensorboard.add_pic(epoch, f'Patch_lv{len(self.MPGAN.netG.gf.feature_list)-i}_feature_map_{j}', img)

            for i in range(len(self.MPGAN.netG.decoder.feature_list)):
                _, C, H, W = self.MPGAN.netG.decoder.feature_list[i].size()
                map_all = (self.MPGAN.netG.decoder.feature_list[i])

                for j in range(C):
                    img = map_all[0][i].unsqueeze(0)
                    self.tensorboard.add_pic(epoch, f'result_lv{i+1}_feature_map_{j}', img)


    def visualize_kernel(self, epoch):
        for name, param in self.MPGAN.netG.gf.named_parameters():
            if ('get motion' in name or 'Patch' in name) and 'weight' in name:
                print(param.size(), name)
                in_channels = param.size()[1]
                out_channels = param.size()[0] 
    
                k_w, k_h = param.size()[3], param.size()[2]
                kernel_all = param.view(-1, 1, k_w, k_h)
                kernel_grid = make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
                self.tensorboard.add_pic(epoch, f'{name}_Weight', kernel_grid)