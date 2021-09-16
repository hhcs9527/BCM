import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models

from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################

class ContentLoss:
	def __init__(self, loss):
		self.criterion = loss
			
	def __call__(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)

class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc().cuda()
			
	def __call__(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
		
class GANLoss(nn.Module):
	def __init__(
			self, use_l1=True, target_real_label=1.0,
			target_fake_label=0.0, tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_l1:
			self.loss = nn.L1Loss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

class DiscLoss:
	def name(self):
		return 'DiscLoss'

	def __init__(self, opt):
		self.adveserial_loss = ContentLoss(nn.BCEWithLogitsLoss().cuda())
		self.lambda_adv = opt.lambda_adv

		
	def get_g_loss(self, net, fake_in, true_in):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		fake_critic = net.forward(fake_in)['encode_feature']

		# Real
		real_critic = net.forward(true_in)['encode_feature']

		# Combined loss
		real = torch.full(real_critic.size(), 1, dtype=torch.float).cuda()
		D_fake = self.adveserial_loss(fake_critic - real_critic, real)

		self.loss_G = self.lambda_adv * D_fake
		return torch.mean(self.loss_G)


	def get_loss(self, net, fake_in, true_in):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		fake_critic = net.forward(fake_in.detach())['encode_feature']

		# Real
		real_critic = net.forward(true_in.detach())['encode_feature']

		real = torch.full(fake_critic.size(), 1, dtype=torch.float).cuda()
		fake = torch.full(fake_critic.size(), 0, dtype=torch.float).cuda()

		# Combined loss
		D_real = self.adveserial_loss(real_critic - fake_critic, real)
		D_fake = self.adveserial_loss(fake_critic - real_critic, fake)

		self.adv_loss = torch.mean(self.lambda_adv * ((D_fake + D_real)/2))

		self.loss_D = self.lambda_adv * ((D_fake + D_real)/2) 
		return torch.mean(self.loss_D)
		

class DiscLossLS(DiscLoss):
	def name(self):
		return 'DiscLossLS'

	def __init__(self, opt, tensor):
		super(DiscLoss, self).__init__(opt, tensor)
		# DiscLoss.initialize(self, opt, tensor)
		self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)
		
	def get_g_loss(self,net, realA, fakeB):
		return DiscLoss.get_g_loss(self,net, realA, fakeB)
		
	def get_loss(self, net, realA, fakeB, realB):
		return DiscLoss.get_loss(self, net, realA, fakeB, realB)
		
		
class DiscLossWGANGP():
	def name(self):
		return 'DiscLossWGAN-GP'

	def __init__(self):
		super(DiscLossWGANGP, self).__init__()
		# DiscLossLS.initialize(self, opt, tensor)
		self.LAMBDA = 10
		
	def get_g_loss(self, net, sharp, deblur):
		# First, G(A) should fake the discriminator
		ef_lv1, ef_lv2, encode_feature, fake_critic, fake_lassify_result = net.forward(deblur)
		return -fake_critic.mean()
		
	def calc_gradient_penalty(self, netD, sharp, deblur):
		alpha = torch.rand(1, 1)
		alpha = alpha.expand(sharp.size())
		alpha = alpha.cuda()

		interpolates = alpha * sharp + ((1 - alpha) * deblur)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)

		ef_lv1, ef_lv2, encode_feature, disc_interpolates, fake_lassify_result = netD.forward(interpolates)

		gradients = autograd.grad(
			outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
			create_graph=True, retain_graph=True, only_inputs=True
		)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
		return gradient_penalty
		
	def get_loss(self, net, sharp, deblur):
		# Fake
		ef_lv1, ef_lv2, encode_feature, fake_critic, fake_lassify_result = net.forward(deblur.detach())
		fake_critic = fake_critic.mean()
		
		# Real
		ef_lv1, ef_lv2, encode_feature, real_critic, fake_lassify_result = net.forward(sharp.detach())
		real_critic = real_critic.mean()

		# Combined loss
		self.loss_D = real_critic - fake_critic
		gradient_penalty = self.calc_gradient_penalty(net, sharp.data, deblur.data)
		return [self.loss_D, gradient_penalty]


def init_loss(opt):
	
	pixel_loss = ContentLoss(nn.MSELoss().cuda())
	content_loss = PerceptualLoss(nn.MSELoss().cuda())

	if opt.gan_type == 'wgan-gp':
		disc_loss = DiscLossWGANGP()
	elif opt.gan_type == 'RaGAN':
		disc_loss = DiscLoss(opt)

	return disc_loss, content_loss, pixel_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss