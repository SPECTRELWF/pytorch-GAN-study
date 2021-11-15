github：https://github.com/SPECTRELWF/pytorch-GAN-study

# 网络结构
最近在疯狂补深度学习一些基本架构的基础，看了一下大佬的GAN的原始论文，说实话一头雾水，不是能看的很懂。推荐B站李宏毅老师的机器学习2021的课程，听完以后明白多了。原始论文中就说了一个generator和一个discriminator的结构，并没有细节的说具体是怎么去定义的，对新手不太友好，参考了Github的Pytorch-Gan-master仓库的代码，做了一下照搬吧，照着敲一边代码就明白了GAN的思想了。网上找了一张稍微好点的网络结构图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2c79072879a440268bb5eb01f65c7b72.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAc3BlY3RyZWx3Zg==,size_18,color_FFFFFF,t_70,g_se,x_16)
因为生成对抗网络需要去度量两个分布之间的距离，原始的GAN并没有一个很好的度量，具体细节可以看李宏毅老师的课。导致GAN的训练会比较困难，而且整个LOSS是基本无变化的，但从肉眼上还是能清楚的看到生成的结果在变好。

# 数据集介绍
使用的是网络上的二次元人脸数据集，数据集链接：
网盘链接：https://pan.baidu.com/s/1MFulwMQJ78U2MCqRUYjkMg
提取码：58v6
其中包含16412张二次元人脸图像，每张图片的分辨率为96*96，

![在这里插入图片描述](https://img-blog.csdnimg.cn/e78d56712f2d4d69a6756b78c156ede2.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/11f91624cfd24b43a846c40386addfa7.png)

只需要在我上一篇文章MNIST手写数字生成的基础上修改一下dataload就行，完整代码可以去github下载
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
from dataloader.face_loader import face_loader
import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('new_images', exist_ok=True)

parser = argparse.ArgumentParser()  # 添加参数

parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=96, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")

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

transforms = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])
# loaddata
train_data = face_loader(transforms)
dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size = opt.batch_size,
    shuffle=True,
)

optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#train
for epoch in range(opt.n_epochs):
    for i ,imgs in enumerate(dataloader):
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
            save_image(gen_imgs.data[:256], "new_images/%d.png" % batches_done, nrow=16, normalize=True)

torch.save(generator.state_dict(),"G.pth")
torch.save(discriminator.state_dict(),"D.pth")
```

# 结果
GAN网络的训练是比较困难的，我设置批大小为1024，训练了两百个epoch，给出一些结果。
第0次迭代：
file:///home/lwf/code/pytorch%E5%AD%A6%E4%B9%A0/GAN/GAN%E4%BA%8C%E6%AC%A1%E5%85%83%E4%BA%BA%E8%84%B8%E7%94%9F%E6%88%90/new_images/0.png![image](https://user-images.githubusercontent.com/51198441/141723678-ea424756-4b2e-4d45-8fe2-af8337a7af98.png)

第1000次迭代：

![在这里插入图片描述](https://img-blog.csdnimg.cn/aa43345f22374144bc0c990ecb35f4cf.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAc3BlY3RyZWx3Zg==,size_20,color_FFFFFF,t_70,g_se,x_16)
这个时候其实就有了人脸的一些基本轮廓了，但细节不好，细节处理是GAN的缺点之一。

第10000次迭代：

file:///home/lwf/code/pytorch%E5%AD%A6%E4%B9%A0/GAN/GAN%E4%BA%8C%E6%AC%A1%E5%85%83%E4%BA%BA%E8%84%B8%E7%94%9F%E6%88%90/new_images/10100.png![image](https://user-images.githubusercontent.com/51198441/141723717-15fe01ac-45a4-4350-85ed-e7c63749bd38.png)

第20000次迭代：

file:///home/lwf/code/pytorch%E5%AD%A6%E4%B9%A0/GAN/GAN%E4%BA%8C%E6%AC%A1%E5%85%83%E4%BA%BA%E8%84%B8%E7%94%9F%E6%88%90/new_images/20100.png![image](https://user-images.githubusercontent.com/51198441/141723748-48748e5c-41bd-476f-8e49-08b871748508.png)

第30000次迭代：
file:///home/lwf/code/pytorch%E5%AD%A6%E4%B9%A0/GAN/GAN%E4%BA%8C%E6%AC%A1%E5%85%83%E4%BA%BA%E8%84%B8%E7%94%9F%E6%88%90/new_images/30100.png![image](https://user-images.githubusercontent.com/51198441/141723762-71c3dcc2-4c6e-4a13-ad10-589c2dbc3f1d.png)

第40000次迭代：

file:///home/lwf/code/pytorch%E5%AD%A6%E4%B9%A0/GAN/GAN%E4%BA%8C%E6%AC%A1%E5%85%83%E4%BA%BA%E8%84%B8%E7%94%9F%E6%88%90/new_images/41000.png![image](https://user-images.githubusercontent.com/51198441/141723779-72ca8c91-ee29-4af8-8f83-8803c4bb3661.png)

从第20000次迭代之后从肉眼上看上去就没什么进步了。
