# import torch related lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# import important lib 
import numpy as np
import os
import math
import argparse
import random
import time
import yaml
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import cv2

# our project related lib
import sub_modules
from datasets import GoProDataset
from calculate_metric import PSNR, SSIM

# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  

class Trainer():
    def __init__(self, args, experiment, METHOD, LEARNING_RATE, EPOCHS, GPU, BATCH_SIZE, IMAGE_SIZE):
        # training part
        self.args = args
        self.Framework = Framework(args) 
        self.config = experiment
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.recurrent = self.args.Recurrent_times

        # for recording benchmark result
        self.best_PSNR = 0
        self.best_SSIM = 0
        # save to json file to see the whole result by different checkpoint
        self.eval_result_dict = {}

        # for training record 
        self.METHOD = METHOD
        self.EPOCHS = EPOCHS
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GPU = GPU
        self.eval_frequency = 100

        # first check
        self.checker()

    def checker(self):
        def file_check():
            '''
                check if missing file for storing result.
            '''
            if os.path.exists(f'./runs/{self.METHOD}/checkpoints/') == False:
                os.mkdir(f'./runs/{self.METHOD}/checkpoints/')  

            if os.path.exists(f'./runs/{self.METHOD}/records/') == False:
                os.mkdir(f'./runs/{self.METHOD}/records/')  

            if os.path.exists(f'./runs/{self.METHOD}/result_picture/') == False:
                os.mkdir(f'./runs/{self.METHOD}/result_picture/')  

            if os.path.exists(f'./runs/{self.METHOD}/feature_maps/') == False:
                os.mkdir(f'./runs/{self.METHOD}/feature_maps/') 

            if os.path.exists(f'./runs/{self.METHOD}/feature_maps/{self.METHOD}_low') == False:
                os.mkdir(f'./runs/{self.METHOD}/feature_maps/{self.METHOD}_low')  

            if os.path.exists(f'./runs/{self.METHOD}/feature_maps/{self.METHOD}_mid') == False:
                os.mkdir(f'./runs/{self.METHOD}/feature_maps/{self.METHOD}_mid')  
                
            if os.path.exists(f'./runs/{self.METHOD}/feature_maps/{self.METHOD}_high') == False:
                os.mkdir(f'./runs/{self.METHOD}/feature_maps/{self.METHOD}_high')  

        def reload_check():
            if self.args.start_epoch > 0:
                    print(f'restart training from {self.args.start_epoch}...')
                    self.Framework.netG.load_state_dict(torch.load(str('./runs/' + self.METHOD + '/checkpoints/' + "/last_G.pkl"))) 
        file_check()
        reload_check()
    
    def get_dataloader(self, state, batch = 1):
        train_dataset = GoProDataset(
            blur_image_files = f'./datas/GoPro/{state}_blur_file.txt',
            sharp_image_files = f'./datas/GoPro/{state}_sharp_file.txt',
            root_dir = './datas/GoPro/',
            crop = True,
            crop_size = self.IMAGE_SIZE,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))
        return DataLoader(train_dataset, batch_size = batch, shuffle=True, num_workers=24, pin_memory=True)    

    def update_best_checkpoints(self, avg_PSNR, avg_SSIM):
        if self.best_PSNR < avg_PSNR and self.best_SSIM < avg_SSIM:
            self.best_PSNR = avg_PSNR 
            self.best_SSIM = avg_SSIM
            torch.save(self.Framework.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "best_G.pkl"))

        elif self.best_PSNR < avg_PSNR:
            self.best_PSNR = avg_PSNR 
            torch.save(self.Framework.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "best_psnr_G.pkl"))

        elif self.best_SSIM < avg_SSIM:
            self.best_SSIM = avg_SSIM
            if self.args.isGAN:
                torch.save(self.Framework.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "best_ssim_G.pkl"))
        torch.save(self.Framework.netG.state_dict(), str('./runs/' + self.METHOD + '/checkpoints/' + "last_G.pkl"))

    def train(self):        
        for epoch in range(self.args.start_epoch, self.EPOCHS):
            self.ep = epoch  
            self.Framework.netG.train()

            # reset training mode to train
            self.args.isTrain = True

            print(f'Training {self.METHOD} at epoch {epoch+1}...., time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

            # get train loader
            train_dataloader = self.get_dataloader('train', self.BATCH_SIZE)

            # for calculateing the metric result
            sub_psnr, sub_ssim = 0, 0
            reblur_psnr, reblur_ssim = 0, 0
            loader_length = len(train_dataloader)

            for iteration, images in enumerate(tqdm(train_dataloader)):
                self.Framework.set_input(images)                     
                self.Framework.optimize_parameters()

                sub_psnr = sub_psnr + PSNR(self.Framework.deblur_dict['deblur'][0], self.Framework.sharp[0])
                sub_ssim = sub_ssim + SSIM(self.Framework.deblur_dict['deblur'][0], self.Framework.sharp[0])
                reblur_psnr = reblur_psnr + PSNR(self.Framework.deblur_dict['reblur']['sharp_reblur'], self.Framework.blur)               
                reblur_ssim = reblur_ssim + SSIM(self.Framework.deblur_dict['reblur']['sharp_reblur'], self.Framework.blur)  

            avg_PSNR = sub_psnr / loader_length
            avg_SSIM = sub_ssim / loader_length
            avg_reblur_PSNR = reblur_psnr / loader_length
            avg_reblur_SSIM = reblur_ssim / loader_length
            self.Framework.update_learning_rate(epoch)

            # for reducing memory 
            del images

            self.update_best_checkpoints(avg_PSNR, avg_SSIM)            
            print(f'epoch: {epoch+1}, psnr: {sub_psnr/len(train_dataloader)}, ssim: {sub_ssim/len(train_dataloader)}, reblur_psnr: {reblur_psnr/len(train_dataloader)}, reblur_ssim: {reblur_ssim/len(train_dataloader)}, time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            
            # eval with checkpoints
            if (epoch + 1) % self.eval_frequency == 0:
                self.eval(checkpoint_G = f'./runs/{self.METHOD}/checkpoints/last_G.pkl')
                torch.save(self.Framework.netG.state_dict(), f'./runs/{self.METHOD}/checkpoints/{self.METHOD}_{epoch + 1}.pkl')

            self.visualize_motion_feature_map(epoch)

    def eval(self, checkpoint_G):
        print(f'evaluating...{checkpoint_G}')
        try:
            print('load {}'.format(checkpoint_G.split('/')[-1][:-4]))
            self.Framework.netG.load_state_dict(torch.load(checkpoint_G))
        except:
            print('no such file')
            return

        # get test loader
        test_dataloader = self.get_dataloader('test', self.BATCH_SIZE)
        
        # for recording
        sub_psnr, sub_ssim, loader_length = 0, 0, len(test_dataloader)

        start_t = time.time()
        with torch.no_grad(): 
            for iteration, images in enumerate(test_dataloader):   
                self.Framework.set_input(images)                     
                self.Framework.test()
                sub_psnr = sub_psnr + PSNR(self.Framework.deblur_image, self.Framework.sharp[0])
                sub_ssim = sub_ssim + SSIM(self.Framework.deblur_image, self.Framework.sharp[0])
                
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

    def eval_all_possible_checkpoints(self):
        checkpoint_list = [str('./runs/' + self.METHOD + '/checkpoints/' + "best_G.pkl"), 
                            str('./runs/' + self.METHOD + '/checkpoints/' + "last_G.pkl"),
                            str('./runs/' + self.METHOD + '/checkpoints/' + "best_psnr_G.pkl"),
                            str('./runs/' + self.METHOD + '/checkpoints/' + "best_ssim_G.pkl")]
        
        for checkpoint in checkpoint_list:
            self.eval(checkpoint)
        
        self.eval_result_dict['Number of trainable parameters'] = sum(p.numel() for p in self.Framework.netG.parameters() if p.requires_grad)
        self.eval_result_dict['experiment_name'] = self.config['experiment_name']
        
        with open(f'./runs/{self.METHOD}/eval_result.json', 'w') as json_file:
            json.dump(self.eval_result_dict, json_file)


    def generate_eval_pic(self):
        def save_images(images, reblur, name):
            save_path = f'./runs/{self.METHOD}/result_picture/{reblur}/'
            if os.path.exists(save_path) == False:
                os.mkdir(save_path)  
            filename = save_path + name
            torchvision.utils.save_image(images, filename)

        print('evaluating...' + self.METHOD)
        checkpoint_G = f'./runs/{self.METHOD}/checkpoints/last_G.pkl'
        
        if os.path.exists(checkpoint_G):
            print('load {}'.format(checkpoint_G.split('/')[-1][:-4]))
            self.Framework.netG.load_state_dict(torch.load(checkpoint_G))
        else:
            return 
        
        test_dataloader = get_dataloader('test')

        self.Framework.netG.opt.isTrain = True
        for iteration, images in enumerate(tqdm(test_dataloader)):
    	    with torch.no_grad():     
                self.Framework.set_input(images)                     
                self.Framework.generate()
                deblur = self.Framework.deblur_image
                reblur = self.Framework.reblur_image             
                save_images(deblur, 'deblur', str(iteration) + '.png')
                save_images(reblur, 'reblur', str(iteration) + '.png')