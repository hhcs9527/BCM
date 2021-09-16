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
from torchvision import transforms, datasets
from datasets import GoProDataset
from datetime import datetime
from DMPHN_zoo import *
from SP_reblur_module import *
from DMPHN_1_2_4_test import *
import time

from calculate_metric import PSNR, SSIM

# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2400)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 6)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMPHN_1_2_4"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize
eval_per_epoch = 100

from tensorboard_vis import MetricCounter
tensorboard = MetricCounter('runs/' + METHOD)

def save_deblur_images(images, iteration, epoch):
    filename = './runs/' + METHOD + '/checkpoints/' + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_deblur.png"
    torchvision.utils.save_image(images, filename)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
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
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print(f'init data folders...{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    deblur_model = DMPHN_1_2_4_reblur().cuda(GPU)    
    reblur_module = blur_understanding_module().cuda(GPU)    
    
    deblur_model.encoder_lv1.apply(weight_init).cuda(GPU)    
    deblur_model.encoder_lv2.apply(weight_init).cuda(GPU)
    deblur_model.encoder_lv3.apply(weight_init).cuda(GPU)

    deblur_model.decoder_lv1.apply(weight_init).cuda(GPU)    
    deblur_model.decoder_lv2.apply(weight_init).cuda(GPU)
    deblur_model.decoder_lv3.apply(weight_init).cuda(GPU)
    
    deblur_model.encoder_lv1_optim = torch.optim.Adam(deblur_model.encoder_lv1.parameters(),lr=LEARNING_RATE)
    deblur_model.encoder_lv1_scheduler = StepLR(deblur_model.encoder_lv1_optim,step_size=1000,gamma=0.1)
    deblur_model.encoder_lv2_optim = torch.optim.Adam(deblur_model.encoder_lv2.parameters(),lr=LEARNING_RATE)
    deblur_model.encoder_lv2_scheduler = StepLR(deblur_model.encoder_lv2_optim,step_size=1000,gamma=0.1)
    deblur_model.encoder_lv3_optim = torch.optim.Adam(deblur_model.encoder_lv3.parameters(),lr=LEARNING_RATE)
    deblur_model.encoder_lv3_scheduler = StepLR(deblur_model.encoder_lv3_optim,step_size=1000,gamma=0.1)

    deblur_model.decoder_lv1_optim = torch.optim.Adam(deblur_model.decoder_lv1.parameters(),lr=LEARNING_RATE)
    deblur_model.decoder_lv1_scheduler = StepLR(deblur_model.decoder_lv1_optim,step_size=1000,gamma=0.1)
    deblur_model.decoder_lv2_optim = torch.optim.Adam(deblur_model.decoder_lv2.parameters(),lr=LEARNING_RATE)
    deblur_model.decoder_lv2_scheduler = StepLR(deblur_model.decoder_lv2_optim,step_size=1000,gamma=0.1)
    deblur_model.decoder_lv3_optim = torch.optim.Adam(deblur_model.decoder_lv3.parameters(),lr=LEARNING_RATE)
    deblur_model.decoder_lv3_scheduler = StepLR(deblur_model.decoder_lv3_optim,step_size=1000,gamma=0.1)

    
    if os.path.exists('./runs/' + METHOD + '/checkpoints/') == False:
        os.system('mkdir ./runs/' + METHOD + '/checkpoints/')  
    
    train_dataset = GoProDataset(
            blur_image_files = './datas/GoPro/train_blur_file.txt',
            sharp_image_files = './datas/GoPro/train_sharp_file.txt',
            root_dir = '/home/hh/Desktop/GoPro/',
            crop = True,
            crop_size = IMAGE_SIZE,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    
    for epoch in range(args.start_epoch, EPOCHS):
        
        print(f'Training ..., time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        sub_psnr = 0
        sub_ssim = 0
        
        for iteration, images in enumerate(train_dataloader):            
            mse = nn.MSELoss().cuda(GPU)          
                
            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)            
            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
            deblur_dict = deblur_model(images_lv1)            

            loss_lv1 = mse(deblur_dict['deblur'], gt)
            loss_reblur = mse(reblur_module(gt, deblur_dict['reblur_filter']), images_lv1)

            loss = loss_lv1 + loss_reblur
            
            deblur_model.encoder_lv1.zero_grad()
            deblur_model.encoder_lv2.zero_grad()
            deblur_model.encoder_lv3.zero_grad()

            deblur_model.decoder_lv1.zero_grad()
            deblur_model.decoder_lv2.zero_grad()
            deblur_model.decoder_lv3.zero_grad()

            loss.backward()

            deblur_model.encoder_lv1_optim.step()
            deblur_model.encoder_lv2_optim.step()
            deblur_model.encoder_lv3_optim.step()

            deblur_model.decoder_lv1_optim.step()
            deblur_model.decoder_lv2_optim.step()
            deblur_model.decoder_lv3_optim.step() 
            
            deblur_model.encoder_lv1_scheduler.step()
            deblur_model.encoder_lv2_scheduler.step()
            deblur_model.encoder_lv3_scheduler.step()

            deblur_model.decoder_lv1_scheduler.step()
            deblur_model.decoder_lv2_scheduler.step()
            deblur_model.decoder_lv3_scheduler.step()       

            sub_psnr += PSNR(deblur_dict['deblur'], gt)
            sub_ssim += SSIM(deblur_dict['deblur'], gt)
            #         
            #if (iteration+1)%50000 == 0:
            #    stop = time.time()
            #    print("epoch:", epoch, "iteration:", iteration+1, "loss:%.4f"%loss.item(), 'time:%.4f'%(stop-start))
            #    start = time.time()

        #print('PSNR : {}, SSIM : {}, time : {}'.format(avg_PSNR, avg_SSIM))
        print(f'epoch: {epoch+1}, psnr: {sub_psnr/len(train_dataloader)}, ssim: {sub_ssim/len(train_dataloader)}, time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        print()
        #print(f'epoch: {epoch}, time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

        #tensorboard.add_metrics_to_tensorboard(epoch, avg_PSNR, avg_SSIM)
        
        if (epoch + 1) % eval_per_epoch == 0:
            print(f'Evaluating ..., time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            
            sub_psnr = torch.tensor(0.0).cuda()
            sub_ssim = torch.tensor(0.0).cuda()
            test_dataset = GoProDataset(
            blur_image_files = './datas/GoPro/test_blur_file.txt',
            sharp_image_files = './datas/GoPro/test_sharp_file.txt',
            root_dir = '/home/hh/Desktop/GoPro/',
            transform = transforms.Compose([
                transforms.ToTensor()
            ]))

            test_dataloader = DataLoader(test_dataset, batch_size = 6, shuffle=False, num_workers=16, pin_memory=True)
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():      
      
                    gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)         
                    images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
                    deblur_dict = deblur_model(images_lv1)
                    sub_psnr += PSNR(deblur_dict['deblur'], gt)
                    sub_ssim += SSIM(deblur_dict['deblur'], gt)
            

            print(f'Eval at {epoch+1}, psnr: {sub_psnr/len(test_dataloader)}, ssim: {sub_ssim/len(test_dataloader)}, time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            print()



    torch.save(deblur_model.encoder_lv1.state_dict(),str('./runs/' + METHOD + '/checkpoints/' + "/encoder_lv1.pkl"))
    torch.save(deblur_model.encoder_lv2.state_dict(),str('./runs/' + METHOD + '/checkpoints/' + "/encoder_lv2.pkl"))
    torch.save(deblur_model.encoder_lv3.state_dict(),str('./runs/' + METHOD + '/checkpoints/' + "/encoder_lv3.pkl"))

    torch.save(deblur_model.decoder_lv1.state_dict(),str('./runs/' + METHOD + '/checkpoints/' + "/decoder_lv1.pkl"))
    torch.save(deblur_model.decoder_lv2.state_dict(),str('./runs/' + METHOD + '/checkpoints/' + "/decoder_lv2.pkl"))
    torch.save(deblur_model.decoder_lv3.state_dict(),str('./runs/' + METHOD + '/checkpoints/' + "/decoder_lv3.pkl"))
                

if __name__ == '__main__':
    main()
    '''
    encoder_lv1 = models.Encoder()
    encoder_lv2 = models.Encoder()    
    encoder_lv3 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()    
    decoder_lv3 = models.Decoder()

    models_li = encoder_lv1, encoder_lv2, encoder_lv3, decoder_lv1, decoder_lv2, decoder_lv3
    total_size = 0
    for model in models_li:
        total_size += sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(total_size/1000000)
    '''
