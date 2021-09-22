import sub_modules

# setting seed for reproduce
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)           
torch.cuda.manual_seed(manualSeed)  

class Framework(nn.Module):
	def __init__(self, opt):
		self.opt = opt
		self.isTrain = opt.isTrain
		self.isGAN = opt.isGAN
		# default 6 for batch
		opt.learning_rate = opt.learning_rate / 6 * opt.batchsize

		# define tensors
		self.isgpu = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
		
		# load/define networks
		self.netG = networks.BRRM_framework(opt)

		if self.isTrain:
			# initialize optimizers			
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.learning_rate)
			self.G_scheduler = StepLR(self.optimizer_G,step_size=1000,gamma=0.1)
												
			# define loss functions
			self.disc_loss, self.content_loss, self.pixel_loss = init_loss(opt)
	
	# Follow the method from DMPHN
	def set_input(self, images):
		self.blur = Variable(images['blur_image'] - 0.5).to(self.isgpu)
		self.sharp = [Variable( images['sharp_image'] - 0.5).to(self.isgpu)]
		
	def set_pic_input(self, images):
		self.blur = Variable(images).to(self.isgpu)

	def forward(self):
		'''
            get 
                1. reblur : self.reproduce_blur (generate a series of sharp image, and average of them should look like input)
                2. deblur : deblur result
		'''
		self.deblur_dict = self.netG(self.blur, self.sharp)

	def test(self):
		'''
			for testing
		'''
		self.deblur_image = self.netG(self.blur, self.sharp)['deblur']

	def generate(self):
		'''
			for demo the result, deblurred image / reblur image
		'''
		self.deblur_image = self.netG(self.blur, self.sharp)['deblur'] + 0.5
		self.reblur_image = self.netG(self.blur, self.sharp)['reblur']['sharp_reblur'] + 0.5

	def backward(self):
		'''
			We have three loss.
			1.	reconstruction blur loss.  
			2.	similar loss.  
			3.	original MSE loss.
		'''
		reconstruct_sharp_loss = self.pixel_loss(self.sharp, deblur)
		reblur_loss = self.pixel_loss(self.blur, self.deblur_dict['reblur']['sharp_reblur']) + self.pixel_loss(self.blur, self.deblur_dict['reblur']['deblur_reblur']) 
		similar_loss = self.pixel_loss(self.deblur_dict['reblur']['sharp_reblur'], self.deblur_dict['reblur']['deblur_reblur'])

		self.loss_G = reconstruct_sharp_loss + similar_loss +  0.1 * reblur_loss
		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_G.zero_grad()	
		self.backward()
		self.optimizer_G.step()

	def update_learning_rate(self, epoch):
		self.G_scheduler.step()