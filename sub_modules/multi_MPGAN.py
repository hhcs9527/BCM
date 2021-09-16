import numpy as np
import torch
import torch.nn as nn
import os, time
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import random
import torch.nn.functional as F
from sub_modules.base_model import BaseModel
import sub_modules.networks as networks
from sub_modules.losses import *
from torchvision.utils import save_image

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
		opt.learning_rate = opt.learning_rate / 6 * opt.batchsize

		# define tensors
		self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'

		# definr lambda for loss
		self.lambda_content = opt.lambda_content
		self.lambda_pixel = opt.lambda_pixel
		self.lambda_adv = opt.lambda_adv
		self.lambda_offset = opt.lambda_offset
		self.G_name = opt.G
		self.reblur_criterion = CharbonnierLoss()

		# load/define networks
		self.netG = networks.define_G(opt, self.isGAN)
		if self.isGAN:
			'''
				net D include 2 dicriminator, image_D for result image, content_D for content feature
				output :
					self.netD = {'image_D' : network, 'content_D' : network}
			'''
			self.netD = {'image_D' : networks.define_image_D(opt)}


		if self.isTrain:
			# initialize optimizers
			if self.isGAN:
				'''
					concept like net D, 
						self.optimizer_D = {'image_D' : torch.optim.Adam(self.netD['image_D'].parameters(),lr = opt.learning_rate), 
						'content_D' : torch.optim.Adam(self.netD['content_D'].parameters(),lr = opt.learning_rate)}

				'''
				self.optimizer_D = {'image_D' : torch.optim.Adam(self.netD['image_D'].parameters(),lr = opt.learning_rate)}
				self.image_D_scheduler = StepLR(self.optimizer_D['image_D'], step_size=1000, gamma=0.1)
				self.criticUpdates = 1
			
			else:
				self.adv_loss, self.class_loss = 0, 0
			
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.learning_rate)
			self.G_scheduler = StepLR(self.optimizer_G,step_size=1000,gamma=0.1)
												
			# define loss functions
			self.disc_loss, self.content_loss, self.pixel_loss = init_loss(opt)
			self.l1_loss = nn.L1Loss().to(self.isgpu)


	def set_input(self, images):
		self.blur = Variable(images['blur_image'] - 0.5).to(self.isgpu)
		self.sharp = [Variable( images['sharp_image'] - 0.5).to(self.isgpu)]
		

	def set_pic_input(self, images):
		self.blur = Variable(images).to(self.isgpu)


	def get_G_loss(self):
		#### L2 loss
		self.reconstruct_sharp_loss = torch.tensor(0.0).to(self.isgpu)		
		for deblur in self.deblur_dict['deblur']:
			self.reconstruct_sharp_loss += self.pixel_loss(self.sharp[0], deblur)

		self.reblur_loss = self.pixel_loss(self.blur, self.deblur_dict['reblur']['sharp_reblur']) 
		self.consistency_loss = self.pixel_loss(self.blur, self.deblur_dict['reblur']['deblur_reblur']) 
		self.similar_loss = self.pixel_loss(self.deblur_dict['reblur']['sharp_reblur'], self.deblur_dict['reblur']['deblur_reblur'])
		return self.reconstruct_sharp_loss + self.similar_loss +  0.1 * (self.reblur_loss + self.consistency_loss)
		#return self.reconstruct_sharp_loss + 0.1 * (self.reblur_loss)

	def get_D_loss(self, which_D):
		'''
			RaGAN loss for the model.
			This funciton returns the discrinator loss for D
		'''	
		image_D = torch.tensor(0.0).to(self.isgpu)
		image_D += self.disc_loss.get_loss(self.netD['image_D'], self.deblur_dict['reblur']['sharp_reblur'].detach(), self.blur)
		image_D += self.disc_loss.get_loss(self.netD['image_D'], self.deblur_dict['reblur']['deblur_reblur'].detach(), self.blur)
		#print('disc', image_D)
		return image_D


	
	def forward(self):
		'''
            input 
                1.  content_feature : encode result of encoder
                2.  alignment_deblur : self.alignment_deblur
                3.  deblur : self.deblur (generate a series of sharp image, and average of them should look like input)
                4.  offset : self.total_offset
		'''
		self.deblur_dict = self.netG(self.blur, self.sharp[0])


	def test(self):
		'''
			Note :
				no backprop gradients
            input 
                1.  content_feature : encode result of encoder
                2.  alignment_deblur : self.alignment_deblur
                3.  deblur : self.deblur (generate a series of sharp image, and average of them should look like input)
                4.  offset : self.total_offset
		'''
		self.deblur_image = self.netG(self.blur, self.sharp[0])['deblur'][0]

	def generate(self):
		'''
			Note :
				no backprop gradients
            input 
                1.  content_feature : encode result of encoder
                2.  alignment_deblur : self.alignment_deblur
                3.  deblur : self.deblur (generate a series of sharp image, and average of them should look like input)
                4.  offset : self.total_offset
		'''
		self.deblur_image = self.netG(self.blur, self.sharp[0])['deblur'][0] + 0.5
		self.reblur_image = self.netG(self.blur, self.sharp[0])['reblur']['sharp_reblur'] + 0.5

	def backward_D(self):
		'''
			backward two discriminator
		'''
		image_D = self.get_D_loss('image_D')
		# backward the D
		image_D.backward()

		self.Discriminator_loss = image_D 

	def backward_G(self):
		'''
			for generator, we have three loss.
			1.	reconstruction blur loss.  
			2.	alignment sharp loss.  
			3.	original MSE loss.
			4.	offset between the latent space
		'''
		if self.isGAN:
			image_D = torch.tensor(0.0).to(self.isgpu)
			image_D += self.disc_loss.get_loss(self.netD['image_D'], self.deblur_dict['reblur']['sharp_reblur'], self.blur)
			image_D += self.disc_loss.get_loss(self.netD['image_D'], self.deblur_dict['reblur']['deblur_reblur'], self.blur)
			self.loss_G_GAN = image_D * self.lambda_adv

		else:
			self.loss_G_GAN = torch.tensor(0.0).to(self.isgpu)
		

		self.loss_G_Content = 0#self.pixel_loss(self.deblur_dict['deblur'][0], self.sharp[0]) * self.lambda_content
		self.loss_G_pixel = self.get_G_loss() * self.lambda_pixel 
		self.loss_G = self.loss_G_pixel #+ self.loss_G_GAN
		self.loss_G.backward()


	def optimize_parameters(self):
		self.forward()

		if self.isGAN:
			for iter_d in range(self.criticUpdates):
				self.optimizer_D['image_D'].zero_grad()
				self.backward_D()
				self.optimizer_D['image_D'].step()

		self.optimizer_G.zero_grad()	
		self.backward_G()
		self.optimizer_G.step()


		#if self.isGAN:
		#	self.adv_loss = self.disc_loss.adv_loss
			
		#else:
		self.adv_loss = 0


	def update_learning_rate(self, epoch):
		if self.isGAN:
			self.image_D_scheduler.step()

		self.G_scheduler.step()