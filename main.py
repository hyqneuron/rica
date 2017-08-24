import torch
from torch.nn import Parameter
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools


"""
(dataset, loader), (dt, lt) = thu.make_data_cifar10(1000)
"""
torchvision_path_cifar10 = '/home/noid/data/torchvision_data/cifar10'
dataset = torchvision.datasets.CIFAR10(
        torchvision_path_cifar10, 
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.469, 0.481, 0.451], std=[0.239,0.245,0.272])
        ]), 
        download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=1000, num_workers=2, pin_memory=True)

data_all = []

for imgs, labels in loader:
    data_all.append(imgs)

data_all = torch.cat(data_all).cuda()



def doit(lambd=1, lrm=1.0, epochs=100):
    global weight, optimizer, epoch, batch, imgs, labels, patches, latents, output, diff

    weight = Parameter(1.0/8.0*torch.Tensor(64,64).normal_().cuda())

    optimizer = torch.optim.RMSprop([weight], 0.001*lrm, momentum=0.9)

    for epoch in xrange(epochs):
        for batch in xrange(data_all.size(0)/1000):
            # capture a few corners
            imgs = data_all[batch*1000:(batch+1)*1000]
            patches = []
            for x,y in itertools.product([3, 10, 17],[3,10,17]):
                patches.append(imgs[:, :, y:y+8, x:x+8])
            patches = Variable(torch.cat(patches).cuda())
            patches = patches.mean(1)
            patches = patches.view(patches.size(0), -1)
            latents= patches.matmul(weight)
            output = latents.matmul(weight.t())
            diff = output - patches
            loss_recon = (diff * diff).mean()
            loss_abs   = latents.abs().mean()
            loss = lambd * loss_recon + loss_abs
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print epoch, loss.data[0], loss_recon.data[0], loss_abs.data[0]
    weight_images = weight.data.t().contiguous().view(64, 1, 8, 8).cpu()
    vutils.save_image(weight_images, 'weight_images_{}.jpg'.format(lambd), normalize=True)

for l in xrange(1, 20):
    doit(l*0.4)

