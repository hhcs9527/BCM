from __future__ import print_function
import argparse
import numpy as np
import torch
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
import tqdm
from albumentations import Compose, CenterCrop, PadIfNeeded
from PIL import Image
from ssim.ssimlib import SSIM
import math

def PSNR(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
	
def get_args():
	parser = argparse.ArgumentParser('Test images')
	parser.add_argument('--ground_truth_img_folder', default = '/home/hh/Desktop/GoPro', help='GoPRO Folder')
	parser.add_argument('--predict_img_folder', default = '/home/hh/Desktop/disk2/MPGAN/test_results/DMPHN_1_2_4_8_test_res', help='GoPRO Folder')
	return parser.parse_args()


def prepare_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def get_gt_image(path):
	dir, filename = os.path.split(path)
	base, seq = os.path.split(dir)
	base, _ = os.path.split(base)
	img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', seq, filename)), cv2.COLOR_BGR2RGB)
	return img

def read_image(path):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.array(Image.open(path))
    #    img2 = np.array(Image.open(predict))
	return img

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def test_image(gt_path, predict_path):
	
	gt = read_image(gt_path)
	predict = read_image(predict_path)

	psnr = PSNR(predict, gt)
	#pilFake = Image.fromarray(predict)
	#pilReal = Image.fromarray(gt)
	ssim =  calculate_ssim(predict, gt)#SSIM(pilFake).cw_ssim_value(pilReal) #SSIM(pilFake, pilReal)  
	return psnr, ssim


def test(files):
	psnr = 0
	ssim = 0

	for gt, predict in tqdm.tqdm(files):
		#print(gt)
		cur_psnr, cur_ssim = test_image(gt, predict)
		print(cur_psnr, cur_ssim)
		#psnr += cur_psnr
		#ssim += cur_ssim
	#L = len(list(files))
	print("PSNR = {}".format(psnr / 1111))
	print("SSIM = {}".format(ssim / 1111))


if __name__ == '__main__':
	args = get_args()
	#with open('config/config.yaml') as cfg:
	#		config = yaml.load(cfg)
	#model = get_generator(config['model'])
	#model.load_state_dict(torch.load(args.weights_path)['model'])
	#model = model.cuda()
	gt_filenames = sorted(glob.glob('/home/hh/Desktop/GoPro' + '/test' + '/**/sharp/*.png', recursive=True))
	#glob.glob(args.ground_truth_img_folder + '/test' + '/**/sharp/*.png', recursive=True)
	#gt_filenames.sort(key=lambda x: x.split("/")[-3])
	predict_filenames = glob.glob(args.predict_img_folder + '/*.png')
	predict_filenames.sort(key=lambda x: int(x.split("/")[-1].split('.')[0]))
	
	test(zip(gt_filenames, predict_filenames))