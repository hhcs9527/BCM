from pacnet.pac import PacConv2d, PacConvTranspose2d
import torch.nn as nn

x = 0
pacconv = nn.Conv2d(channel*2, channel*4, kernel_size=3, padding=1)