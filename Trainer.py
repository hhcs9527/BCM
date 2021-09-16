import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms, datasets
#from one_to_all_datasets import GoProDataset
from datasets import GoProDataset
import time
import yaml
import json
from datetime import datetime
import sub_modules.networks as networks
from PIL import Image
#from sub_modules.get_GAN import MPGAN
from sub_modules.multi_MPGAN import MPGAN
from calculate_metric import PSNR, SSIM
from tensorboard_vis import MetricCounter
from torchvision.utils import make_grid
from skimage import measure
from psnr_ssim import metric_calculator
from tqdm import tqdm
import cv2


# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  


class Trainer():
    def __init__(self, args, experiment, METHOD, LEARNING_RATE, EPOCHS, GPU, BATCH_SIZE, IMAGE_SIZE):
        self.args = args
        self.MPGAN = MPGAN(args) 
        self.best_PSNR = 0
        self.best_SSIM = 0
        self.tensorboard = MetricCounter('runs/' + METHOD)
        # save to json file to see the whole result by different checkpoint
        self.eval_result_dict = {}
        self.config = experiment
        self.METHOD = METHOD
        self.EPOCHS = EPOCHS
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GPU = GPU
        self.eval_frequency = 100
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.recurrent = self.args.Recurrent_times

    def train(self):        
        if os.path.exists(f'./runs/{self.METHOD}/checkpoints/') == False:
            os.mkdir(f'./runs/{self.METHOD}/checkpoints/')  

        if os.path.exists(f'./runs/{self.METHOD}/records/') == False:
            os.mkdir(f'./runs/{self.METHOD}/records/')  

        if os.path.exists(f'./runs/{self.METHOD}/result_picture/') == False:
            os.mkdir(f'./runs/{self.METHOD}/result_picture/')  

        if os.path.exists(f'./runs/{self.METHOD}/feature_maps/') == False:
            os.mkdir(f'./runs/{self.METHOD}/feature_maps/')  

        if self.args.start_epoch > 0:
                print(f'restart training from {self.args.start_epoch}...')
                self.MPGAN.netG.load_state_dict(torch.load(str('./runs/' + self.METHOD + '/checkpoints/' + "/last_G.pkl"))) 

        for epoch in range(self.args.start_epoch, self.EPOCHS):
            self.ep = epoch  
            self.MPGAN.netG.train()
            if (epoch % self.eval_frequency == 0 and epoch > 0):
                self.MPGAN.netG.load_state_dict(torch.load(str('./runs/' + self.METHOD + '/checkpoints/' + "/last_G.pkl"))) 

            print(f'Training {self.METHOD} at epoch {epoch+1}...., time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            train_dataset = GoProDataset(
                blur_image_files = './datas/GoPro/train_blur_file.txt',
                sharp_image_files = './datas/GoPro/train_sharp_file.txt',
                root_dir = './datas/GoPro/',
                crop = True,
                crop_size = self.IMAGE_SIZE,
                transform = transforms.Compose([
                    transforms.ToTensor()
                    ]))

            train_dataloader = DataLoader(train_dataset, batch_size = self.BATCH_SIZE, shuffle=True, num_workers=24, pin_memory=True)
            
            start = 0

            adv_Dloss_list = []
            adv_Gloss_list = []
            content_loss_list = []
            reblur_loss = 0

            sub_psnr, sub_ssim, pix_loss, reblur_loss = 0, 0, 0, 0
            reblur_psnr, reblur_ssim = 0, 0
            self.args.isTrain = True
            loader_length = len(train_dataloader)

            for iteration, images in enumerate(tqdm(train_dataloader)):

                self.MPGAN.set_input(images)                     
                self.MPGAN.optimize_parameters()

                sub_psnr = sub_psnr + PSNR(self.MPGAN.deblur_dict['deblur'][0], self.MPGAN.sharp[0])
                sub_ssim = sub_ssim + SSIM(self.MPGAN.deblur_dict['deblur'][0], self.MPGAN.sharp[0])
                reblur_psnr = reblur_psnr + PSNR(self.MPGAN.deblur_dict['reblur']['sharp_reblur'], self.MPGAN.blur)               
                reblur_ssim = reblur_ssim + SSIM(self.MPGAN.deblur_dict['reblur']['sharp_reblur'], self.MPGAN.blur)  

            avg_PSNR = sub_psnr / loader_length
            avg_SSIM = sub_ssim / loader_length
            avg_reblur_PSNR = reblur_psnr / loader_length
            avg_reblur_SSIM = reblur_ssim / loader_length
            self.MPGAN.update_learning_rate(epoch)
            del images
            
            if self.best_PSNR < avg_PSNR and self.best_SSIM < avg_SSIM:
                self.best_PSNR = avg_PSNR 
                self.best_SSIM = avg_SSIM
                torch.save(self.MPGAN.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "best_G.pkl"))

            elif self.best_PSNR < avg_PSNR:
                self.best_PSNR = avg_PSNR 
                torch.save(self.MPGAN.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "best_psnr_G.pkl"))

            elif self.best_SSIM < avg_SSIM:
                self.best_SSIM = avg_SSIM
                if self.args.isGAN:
                    torch.save(self.MPGAN.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "best_ssim_G.pkl"))
            
            print(f'epoch: {epoch+1}, psnr: {sub_psnr/len(train_dataloader)}, ssim: {sub_ssim/len(train_dataloader)}, reblur_psnr: {reblur_psnr/len(train_dataloader)}, reblur_ssim: {reblur_ssim/len(train_dataloader)}, time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            torch.save(self.MPGAN.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "last_G.pkl"))
            
            if self.MPGAN.isGAN:
                torch.save(self.MPGAN.netD['image_D'].state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "last_D.pkl"))
            print()
            
            if (epoch + 1) % self.eval_frequency == 0:
                self.eval(checkpoint_G = f'./runs/{self.METHOD}/checkpoints/last_G.pkl')
                torch.save(self.MPGAN.netG.state_dict(), f'./runs/{self.METHOD}/checkpoints/{self.METHOD}_{epoch + 1}.pkl')
            self.visualize_motion_feature_map(epoch)

    def eval_checkpoints(self):
        checkpoint_list = [str('./runs/' + self.METHOD + '/checkpoints/' + "best_G.pkl"), 
                            str('./runs/' + self.METHOD + '/checkpoints/' + "last_G.pkl"),
                            str('./runs/' + self.METHOD + '/checkpoints/' + "best_psnr_G.pkl"),
                            str('./runs/' + self.METHOD + '/checkpoints/' + "best_ssim_G.pkl")]
        
        for checkpoint in checkpoint_list:
            self.eval(checkpoint)
        
        self.eval_result_dict['Number of trainable parameters'] = sum(p.numel() for p in self.MPGAN.netG.parameters() if p.requires_grad)
        self.eval_result_dict['experiment_name'] = self.config['experiment_name']
        
        with open(f'./runs/{self.METHOD}/eval_result.json', 'w') as json_file:
            json.dump(self.eval_result_dict, json_file)


    def eval(self, checkpoint_G):
        print(f'evaluating...{checkpoint_G}')
        if os.path.exists(checkpoint_G):
            print('load {}'.format(checkpoint_G.split('/')[-1][:-4]))
            self.MPGAN.netG.load_state_dict(torch.load(checkpoint_G))
        else:
            print('no such file')
            return 

        test_dataset = GoProDataset(
        blur_image_files = './datas/GoPro/test_blur_file.txt',
        sharp_image_files = './datas/GoPro/test_sharp_file.txt',
        root_dir = '/home/hh/Desktop/GoPro/',
        transform = transforms.Compose([
            transforms.ToTensor()
        ]))

        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=24, pin_memory=True)
        sub_psnr, sub_ssim, loader_length = 0, 0, len(test_dataloader)

        start_t = time.time()
        with torch.no_grad(): 
            for iteration, images in enumerate(test_dataloader):   
                self.MPGAN.set_input(images)                     
                self.MPGAN.test()
                sub_psnr = sub_psnr + PSNR(self.MPGAN.deblur_image, self.MPGAN.sharp[0])
                sub_ssim = sub_ssim + SSIM(self.MPGAN.deblur_image, self.MPGAN.sharp[0])
                
        stop_t = time.time()
        del images
        avg_PSNR = sub_psnr / loader_length
        avg_SSIM = sub_ssim / loader_length

        print('RunTime:%.4f'%(stop_t-start_t))    
        print(f'Eval result PSNR : {avg_PSNR}, SSIM : {avg_SSIM}, time : {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        print()      

        if not self.args.isTrain :
            self.eval_result_dict[checkpoint_G.split('/')[-1][:-4]] = {'PSNR' : avg_PSNR, 'SSIM' : avg_SSIM.item()}
        else:
            eval_dict = {'Test PSNR' : avg_PSNR, 'Test SSIM' : avg_SSIM.item()}
            with open(f'./runs/{self.METHOD}/records/{self.METHOD}_{self.ep + 1}_best_psnr_result.json', 'w') as json_file:
                json.dump(eval_dict, json_file)

    def visualize_motion_feature_map(self, epoch = 3000):

        def normalize_img(x):
            return (x - x.min())/(x.max() - x.min())

        print(f'visualizing {self.METHOD} feature map...')
        visual_blur = transforms.ToTensor()(Image.open('/home/hh/Desktop/GoPro/test/GOPR0384_11_05/blur/004003.png').convert('RGB')).unsqueeze(0).to(self.device)
        visual_sharp = transforms.ToTensor()(Image.open('/home/hh/Desktop/GoPro/test/GOPR0384_11_00/sharp/000001.png').convert('RGB')).unsqueeze(0).to(self.device)
        self.MPGAN.netG.load_state_dict(torch.load(f'./runs/{self.METHOD}/checkpoints/last_G.pkl'))
        #self.MPGAN.netG.load_state_dict(torch.load(f'./runs/{self.METHOD}/checkpoints/{self.METHOD}_{epoch}.pkl'))
        images = {'blur_image' : visual_blur, 'sharp_image': visual_sharp}
        files = self.METHOD
        '''
            input : 
                store the attentioned layer, this function support storing a single image
                feature_dict : {lv1_low, lv1_mid, lv1_high....}
            
            output :
                directory path = f'./runs/{self.METHOD}/feature_maps/'
                file name = directory path + feature_list[key]
        '''

        if os.path.exists(f'./runs/{self.METHOD}/feature_maps/{files}_low') == False:
            os.mkdir(f'./runs/{self.METHOD}/feature_maps/{files}_low')  
        if os.path.exists(f'./runs/{self.METHOD}/feature_maps/{files}_mid') == False:
            os.mkdir(f'./runs/{self.METHOD}/feature_maps/{files}_mid')  
        if os.path.exists(f'./runs/{self.METHOD}/feature_maps/{files}_high') == False:
            os.mkdir(f'./runs/{self.METHOD}/feature_maps/{files}_high')  

        self.args.isTrain = False
        directory_path = f'./runs/{self.METHOD}/feature_maps'
        with torch.no_grad():     
            self.MPGAN.set_input(images)      
            self.MPGAN.netG.eval()         
            self.MPGAN.test()
            for key, feature_maps in self.MPGAN.netG.DMPHN.feature_dict.items():
                _, C, H, W = feature_maps.size()
                feature_maps_mean = torch.mean(feature_maps, dim = 1)
                images = feature_maps_mean.permute(1,2,0).cpu().numpy() # cv2 -> H,W,C
                images = normalize_img(images)
                file_name = f'{directory_path}/{files}_{key[4:]}/{key}_feature_map_{epoch}.png'
                heatmapimg = np.array(images * 255, dtype = np.uint8)
                # color space, COLORMAP_WINTER, COLORMAP_SPRING, cv2.COLORMAP_SUMMER, COLORMAP_BONE, COLORMAP_JET
                heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
                #heatmap = cv2.addWeighted(cv2.imread('/home/hh/Desktop/GoPro/test/GOPR0384_11_00/blur/000083.png'), 0.5, heatmap, 0.5, 0)
                cv2.imwrite(file_name, heatmap)


    def visualize_level_output(self):
        print(f'visualizing {self.METHOD} level_output...')
        visual_blur = transforms.ToTensor()(Image.open('/home/hh/Desktop/GoPro/test/GOPR0384_11_05/blur/004003.png').convert('RGB')).unsqueeze(0).to(self.device)
        visual_sharp = transforms.ToTensor()(Image.open('/home/hh/Desktop/GoPro/test/GOPR0384_11_00/sharp/000001.png').convert('RGB')).unsqueeze(0).to(self.device)
        #self.MPGAN.netG.load_state_dict(torch.load(f'./runs/{self.METHOD}/checkpoints/last_G.pkl'))
        self.MPGAN.netG.load_state_dict(torch.load(f'./runs/{self.METHOD}/checkpoints/last_G.pkl'))

        images = {'blur_image' : visual_blur, 'sharp_image': visual_sharp}
        files = self.METHOD
        '''
            input : 
                store the attentioned layer, this function support storing a single image
                feature_dict : {lv1, lv2...}
            
            output :
                directory path = f'./runs/{self.METHOD}/level_output/'
                file name = directory path + feature_list[key]
        '''

        if os.path.exists(f'./runs/{self.METHOD}/level_output') == False:
            os.mkdir(f'./runs/{self.METHOD}/level_output')  

        self.args.isTrain = False
        directory_path = f'./runs/{self.METHOD}/level_output'
        with torch.no_grad():     
            self.MPGAN.set_input(images)               
            self.MPGAN.test()
            for key, feature_maps in self.MPGAN.netG.DMPHN.level_dict.items():
                filename = f'{key}_outputs.png'
                torchvision.utils.save_image(feature_maps + 0.5, os.path.join(directory_path, filename))


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


    def save_images(self, images, reblur, name):
        save_path = f'./runs/{self.METHOD}/result_picture/{reblur}/'
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)  
        filename = save_path + name
        torchvision.utils.save_image(images, filename)

    def generate_eval_pic(self):
        checkpoint_G = f'./runs/{self.METHOD}/checkpoints/last_G.pkl'#str('./runs/' + self.METHOD + '/checkpoints/' + "best_psnr_G.pkl")
        print('evaluating...' + self.METHOD)
        
        if os.path.exists(checkpoint_G):
            print('load {}'.format(checkpoint_G.split('/')[-1][:-4]))
            self.MPGAN.netG.load_state_dict(torch.load(checkpoint_G))
        else:
            return 
        

        test_dataset = GoProDataset(
        blur_image_files = './datas/GoPro/test_blur_file.txt',
        sharp_image_files = './datas/GoPro/test_sharp_file.txt',
        root_dir = '/home/hh/Desktop/GoPro/',
        transform = transforms.Compose([
            transforms.ToTensor()
        ]))

        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=16, pin_memory=True)
        self.MPGAN.netG.opt.isTrain = True
        for iteration, images in enumerate(tqdm(test_dataloader)):
    	    with torch.no_grad():     
                self.MPGAN.set_input(images)                     
                self.MPGAN.generate()
                deblur = self.MPGAN.deblur_image
                reblur = self.MPGAN.reblur_image             
                self.save_images(deblur, 'deblur', str(iteration) + '.png')
                self.save_images(reblur, 'reblur', str(iteration) + '.png')
        
        #self.lib_calculate_metric()

    def lib_calculate_metric(self):
        PSNR_list = []
        SSIM_list = []
        path = './runs/' + self.METHOD + '/result_picture/'
        metric_calculator('deblur', f'./runs/{self.METHOD}/result_picture/deblur')
        metric_calculator('reblur', f'./runs/{self.METHOD}/result_picture/reblur')
        
        

if __name__ == '__main__':
    # for usging shell to run different config
    default_path = 'experiment_config/16/Patch_Motion_SFT_fusion_c16.yaml'
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("-conf","--config",type=str, default = default_path)
    config_args = config_parser.parse_args()

    with open(config_args.config) as f:
        experiment = yaml.load(f, Loader = yaml.FullLoader)

    args = config(experiment)              

    #Hyper Parameters
    METHOD = experiment['experiment_name']
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    GPU = args.gpu
    BATCH_SIZE = args.batchsize
    IMAGE_SIZE = args.imagesize

    args.isTrain = False
    MP = Trainer(args, experiment, METHOD, LEARNING_RATE, EPOCHS, GPU, BATCH_SIZE, IMAGE_SIZE)
    print('Running ... {} epochs !!! time {}'.format(args.epochs, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    if args.isTrain:
        #model = MP.MPGAN.netG
        #print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000)
        MP.train()
        #MP.eval_checkpoints()
    else:
        print('saving')
        MP.generate_eval_pic()
        #MP.lib_calculate_metric()