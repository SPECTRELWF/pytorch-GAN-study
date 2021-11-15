github：https://github.com/SPECTRELWF/pytorch-GAN-study
个人主页：liuweifeng.top:8090

# 网络结构
最近在疯狂补深度学习一些基本架构的基础，看了一下大佬的GAN的原始论文，说实话一头雾水，不是能看的很懂。推荐B站李宏毅老师的机器学习2021的课程，听完以后明白多了。原始论文中就说了一个generator和一个discriminator的结构，并没有细节的说具体是怎么去定义的，对新手不太友好，参考了Github的Pytorch-Gan-master仓库的代码，做了一下照搬吧，照着敲一边代码就明白了GAN的思想了。网上找了一张稍微好点的网络结构图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2c79072879a440268bb5eb01f65c7b72.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAc3BlY3RyZWx3Zg==,size_18,color_FFFFFF,t_70,g_se,x_16)

因为生成对抗网络需要去度量两个分布之间的距离，原始的GAN并没有一个很好的度量，具体细节可以看李宏毅老师的课。导致GAN的训练会比较困难，而且整个LOSS是基本无变化的，但从肉眼上还是能清楚的看到生成的结果在变好。

# 数据集介绍
使用的是经典的MNIST数据集，后期会拿一些人脸数据集来做实验。

# generator

```python
# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            * block(opt.latent_dim,128,normalize=False),
            * block(128,256),
            * block(256,512),
            * block(512,1024),
            nn.Linear(1024,int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self,z):
        img = self.model(z)
        img = img.view(img.size(0),*image_shape)
        return img
```

# discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)),512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        img_flat = img.view(img.size(0),-1)
        validity = self.model(img_flat)

        return validity
```

完整代码：

```python
# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/14 下午3:05

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('new_images', exist_ok=True)

parser = argparse.ArgumentParser()  # 添加参数

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

opt = parser.parse_args()
print(opt)

image_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            * block(opt.latent_dim,128,normalize=False),
            * block(128,256),
            * block(256,512),
            * block(512,1024),
            nn.Linear(1024,int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self,z):
        img = self.model(z)
        img = img.view(img.size(0),*image_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)),512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        img_flat = img.view(img.size(0),-1)
        validity = self.model(img_flat)

        return validity

# loss

adversarial_loss = torch.nn.BCELoss()

#初始化G和D
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# loaddata
os.makedirs("data/mnist",exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train = True,
        download=True,
        transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
        ])
    ),
    batch_size=opt.batch_size,
    shuffle = True
)

optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#train
for epoch in range(opt.n_epochs):
    for i ,(imgs,_) in enumerate(dataloader):
        valid = Variable(Tensor(imgs.size(0),1).fill_(1.0),requires_grad = False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        optimizer_G.zero_grad()

        z = Variable(Tensor(np.random.normal(0,1,(imgs.shape[0],opt.latent_dim))))

        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs),valid)

        g_loss.backward()
        optimizer_G.step()

        #train Discriminator

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs),valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),fake)

        d_loss = (real_loss+fake_loss)/2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:1024], "new_images/%d.png" % batches_done, nrow=32, normalize=True)

torch.save(generator.state_dict(),"G.pth")
torch.save(discriminator.state_dict(),"D.pth")
```

# 结果
GAN网络的训练是比较困难的，我设置批大小为1024，训练了两百个epoch，给出一些结果。
第0次迭代：
![在这里插入图片描述](https://img-blog.csdnimg.cn/fd2adb9eea6b47fcbd5d0ad91014e733.png)

基本上就是纯纯噪声了，初始数据采样来源于标准正态分布。

第400次迭代：
![在这里插入图片描述](https://img-blog.csdnimg.cn/49d7001a73e24954b2f873a438378eb7.png)

第10000次迭代：
![在这里插入图片描述](https://img-blog.csdnimg.cn/653856dd4c974c9ca60922beeb70a656.png)

第186800次迭代：
![在这里插入图片描述](https://img-blog.csdnimg.cn/f630c0c1c1ca4d50a67395ac67b546b3.png)

此时就已经基本有了数字的样子了
