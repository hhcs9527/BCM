import torch
from torch.optim.lr_scheduler import MultiplicativeLR
step = (0.1 - 0.01)/(95)

lmbda = lambda epoch: 1 if epoch <= 5 else 1 - ((epoch - 5) * step / 0.1)

optimizer = torch.optim.SGD([torch.rand((2,2), requires_grad=True)], lr = 0.1)

def lr_decay(epoch):
    if epoch < 5:
        return 1
    else:
        #step = (0.1)/(100 - 5)
        return 1 - ((epoch-5) * step / 0.1)


for epoch in range(100):
    print(lr_decay(epoch))

scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
for epoch in range(100):
    scheduler.step()
    #print(optimizer.param_groups[0]['lr'])