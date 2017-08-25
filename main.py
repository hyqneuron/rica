import torch
from torch.nn import Parameter
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools

"""
Reproduces Reconstruction ICA with PyTorch

1. Modify `torchvision_path_cifar10` to your cifar10 path, or just any folder (it will download dataset automatically)
2. If you do not have a GPU, set `use_gpu=False`. It's going to take more than a few minutes.. If you want to speed
   things up a bit:
   - change lambdas to just [2.4], this runs the script with a single lambda value only, and gives decent result
   - maybe reduce num_epochs to 100
   - if you want to run all the lambda values
     - reduce patch_size to 8, which is probably 2x faster than 16
     - reduce num_epochs to 40
"""

use_gpu    = False              # if to use GPU
num_epochs = 200                # how long each lambda runs, 200 is probably overkill
num_steps  = 20                 # how many lambdas to try
patch_size = 16                 # patch size to extract, 16 is max
weight_size= patch_size**2      # weight size is number of pixels in a patch (do not change)
num_filters = weight_size       # complete-ICA has same number of filters as there are pixels
lambdas = [l*0.4 for l in xrange(1,num_steps)] # the lambda values will be tried one by one
torchvision_path_cifar10 = '/home/noid/data/torchvision_data/cifar10'


def maybe_gpu(data):
    return data.cuda() if use_gpu else data


# use cifar10 as dataset
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
data_all = torch.cat(data_all)      # merge into single tensor
data_all = data_all.mean(1)         # make black-white
data_all = maybe_gpu(data_all)



def doit(lambd=1, epochs=num_epochs):
    weight    = Parameter(maybe_gpu(1.0/patch_size*torch.Tensor(weight_size,num_filters).normal_()))
    optimizer = torch.optim.RMSprop([weight], 0.001, momentum=0.9)

    for epoch in xrange(epochs):
        for batch in xrange(data_all.size(0)/1000):
            # select batch
            imgs = data_all[batch*1000:(batch+1)*1000]
            # capture a few patches
            patches = []
            for x,y in itertools.product([0, 8, 16],[0,8,16]):
                patches.append(imgs[:, y:y+patch_size, x:x+patch_size])
            patches = Variable(maybe_gpu(torch.cat(patches)))
            patches = patches.view(patches.size(0), -1)
            latents= patches.matmul(weight)
            output = latents.matmul(weight.t())
            diff = output - patches
            loss_recon = (diff * diff).mean()
            loss_latent= latents.abs().mean()
            loss = lambd * loss_recon + loss_latent
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print epoch, loss.data[0], loss_recon.data[0], loss_latent.data[0]
    weight_images = weight.data.t().contiguous().view(num_filters, 1, patch_size, patch_size).cpu()
    vutils.save_image(weight_images, 'rica_weight_images_{}.jpg'.format(lambd), nrow=patch_size, normalize=True)
    print 'Finished lambda={}'.format(lambd)


for l in lambdas:
    doit(l)

