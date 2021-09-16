from tensorboardX import SummaryWriter
import numpy as np


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)

    def add_metrics_to_tensorboard(self, epoch, psnr, ssim):
        self.writer.add_scalar('PSNR/train', psnr, epoch)
        self.writer.add_scalar('SSIM/train', ssim, epoch)

    def add_Loss_to_tensorboard(self, epoch, loss_name, loss):
        self.writer.add_scalar(loss_name + '/train', loss, epoch)

    def add_pic(self, epoch, img_name, img):
        self.writer.add_image(img_name, img, global_step=epoch, dataformats='CHW')
