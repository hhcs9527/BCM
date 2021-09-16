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
import time
from PIL import Image

from calculate_metric import PSNR, SSIM

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2600)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMPHN_1_2_4"
SAMPLE_DIR = "test_samples"
EXPDIR = "DMPHN_1_2_4_test_res"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_images(images, name):
    filename = './test_results/' + EXPDIR + "/" + name
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
    print("init data folders")

    encoder_lv1 = models.Encoder().apply(weight_init).cuda()
    encoder_lv2 = models.Encoder().apply(weight_init).cuda()
    encoder_lv3 = models.Encoder().apply(weight_init).cuda()

    decoder_lv1 = models.Decoder().apply(weight_init).cuda()
    decoder_lv2 = models.Decoder().apply(weight_init).cuda()
    decoder_lv3 = models.Decoder().apply(weight_init).cuda()

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        decoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
        print("load decoder_lv3 success")

    
    if os.path.exists('./test_results/' + EXPDIR) == False:
        os.system('mkdir ./test_results/' + EXPDIR)     
            
    iteration = 0.0
    test_time = 0.0
    PSNR_list = []
    SSIM_list = []

    #for images_name in os.listdir(SAMPLE_DIR):

    test_dataset = GoProDataset(
    blur_image_files = './datas/GoPro/test_blur_file.txt',
    sharp_image_files = './datas/GoPro/test_sharp_file.txt',
    root_dir = '/home/hh/Desktop/GoPro/',
    transform = transforms.Compose([
        transforms.ToTensor()
    ]))

    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, pin_memory=True)
    for iteration, images in enumerate(test_dataloader):
    	with torch.no_grad():     
            print(iteration)        
            #images_lv1 = transforms.ToTensor()(Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB'))
            #images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)
            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU) 
            start = time.time()           
            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

            feature_lv3_1 = encoder_lv3(images_lv3_1)
            feature_lv3_2 = encoder_lv3(images_lv3_2)
            feature_lv3_3 = encoder_lv3(images_lv3_3)
            feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)

            feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)
            #torchvision.utils.save_image(images, filename)
            #print(os.path.exists(f'/home/hh/Desktop/disk2/MPGAN/runs/DMPHN/'))
            torchvision.utils.save_image(deblur_image.data + 0.5, f'/home/hh/Desktop/disk2/MPGAN/runs/DMPHN/result_picture/deblur/{iteration}.png') 
            #iteration += 1
    
    #print('PSNR : {}, SSIM : {}'.format(sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
            
if __name__ == '__main__':
    main()

        

        
