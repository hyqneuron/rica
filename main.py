import torch
from torch.nn import Parameter
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools

"""
1. Modify `torchvision_path_cifar10` to your cifar10 path, or just any folder (it will download dataset automatically)
2. remove the 3 occurences of `.cuda()` if you don't have a GPU. It'll take 5x more time, but still a few minutes only
   if your CPU isn't too bad.
"""

num_epochs = 60
num_steps  = 20
torchvision_path_cifar10 = '/home/noid/data/torchvision_data/cifar10'

dataset = torchvision.datasets.CIFAR10(
        torchvision_path_cifar10, 
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.469, 0.481, 0.451], std=[0.239,0.245,0.272])
            # normalize to 0-mean, unit-variance
        ]), 
        download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=1000, num_workers=2, pin_memory=True)

# load the entire dataset into a single Tensor, this speeds things up quite a bit
data_all = []
for imgs, labels in loader:
    data_all.append(imgs)
data_all = torch.cat(data_all).cuda()



def doit(lambd=1, lrm=1.0, epochs=num_epochs, ica=True):
    # ica=True: reconstruction-ICA, ica=False: reconstruction-PCA
    weight    = Parameter(1.0/8.0*torch.Tensor(64,64).normal_().cuda())
    optimizer = torch.optim.RMSprop([weight], 0.001*lrm, momentum=0.9)

    for epoch in xrange(epochs):
        for batch in xrange(data_all.size(0)/1000):
            # select batch
            imgs = data_all[batch*1000:(batch+1)*1000]
            # capture a few patches
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
            loss_latent= latents.abs().mean() if ica else (latents*latents).mean() # PCA
            loss = lambd * loss_recon + loss_latent
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print epoch, loss.data[0], loss_recon.data[0], loss_latent.data[0]
    weight_images = weight.data.t().contiguous().view(64, 1, 8, 8).cpu()
    vutils.save_image(weight_images, 'r{}_weight_images_{}.jpg'.format('ica' if ica else 'pca', lambd), normalize=True)
    print 'Finished lambda={}'.format(lambd)


for l in xrange(1, num_steps):
    doit(l*0.4, ica=True)
    #doit(l*0.4, ica=False) # do this if you want to try reconstruction-PCA. It doesn't work well.

