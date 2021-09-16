import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random

from sub_modules.base_model import BaseModel
import sub_modules.networks as networks
from sub_modules.losses import init_loss

# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  

class MPGAN(nn.Module):
	def name(self):
		return 'ConditionalGANModel'

	def __init__(self, opt):
		super(MPGAN, self).__init__()
		self.opt = opt
		self.isTrain = opt.isTrain
		self.isGAN = opt.isGAN

		# define tensors
		self.isgpu = 'cuda' if torch.cuda.is_available() else 'cpu'

		# definr lambda for loss
		self.lambda_content = opt.lambda_content
		self.lambda_pixel = opt.lambda_pixel
		self.lambda_adv = opt.lambda_adv

		# load/define networks
		self.netG = networks.define_G(opt, self.isGAN)
		if self.isGAN:
			self.netD = networks.define_D(opt)
		
		if self.Recurrent_times > 1:
			self.alignment_module = 5

		if self.isTrain:
			# initialize optimizers
			if self.isGAN:
				self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr = opt.learning_rate)
				self.D_scheduler = StepLR(self.optimizer_D,step_size=1000,gamma=0.1)
				self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
			
			else:
				self.adv_loss, self.class_loss = 0, 0

			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.learning_rate)
			self.G_scheduler = StepLR(self.optimizer_G,step_size=1000,gamma=0.1)
												
			# define loss functions
			self.disc_loss, self.content_loss, self.pixel_loss = init_loss(opt)

	def set_input(self, images):
		self.blur = Variable(images['blur_image'] - 0.5).to(self.isgpu)
		self.sharp = Variable(images['sharp_image'] - 0.5).to(self.isgpu)

	def forward(self):
		'''
			self.netG(self.blur) is a dictionary contain reconstruct blur + deblur image
		'''
		self.deblur = self.netG(self.blur)

	# no backprop gradients
	def test(self):
		'''
			self.netG(self.blur) is a dictionary contain reconstruct blur + deblur image
		'''
		self.deblur = self.netG(self.blur)

	def backward_D(self):
		self.deblur_D = self.deblur.detach()
		self.loss_D = self.disc_loss.get_loss(self.netD, self.deblur_D, self.sharp)

		self.loss_D.backward()

	def backward_G(self):
		if self.isGAN:
			self.loss_G_GAN = self.disc_loss.get_g_loss(self.netD, self.deblur, self.sharp)
		else:
			self.loss_G_GAN = 0

		self.loss_G_Content = self.content_loss(self.deblur, self.sharp) * self.lambda_content

		self.loss_G_pixel = self.pixel_loss(self.deblur, self.sharp) * self.lambda_pixel
		#self.loss_G_pixel += self.pixel_loss(self.blur, self.blur_reconstruct) * self.lambda_pixel

		KL_loss_sharp = 0 
		KL_loss_blur = 0 
		self.loss_G_KL = (KL_loss_blur + KL_loss_sharp)

		self.loss_G = self.loss_G_Content + self.loss_G_pixel + self.loss_G_GAN + self.loss_G_KL

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()
		if self.isGAN:
			for iter_d in range(self.criticUpdates):
				self.optimizer_D.zero_grad()
				self.backward_D()
				self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

		if self.isGAN:
			self.adv_loss = self.disc_loss.adv_loss + self.loss_G_GAN 
		
		else:
			self.adv_loss = 0
			self.class_loss = 0


	def update_learning_rate(self, epoch):
		if self.isGAN:
			self.D_scheduler.step(epoch)

		self.G_scheduler.step(epoch)