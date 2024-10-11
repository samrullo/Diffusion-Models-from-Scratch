import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torch.optim as optim
from models.ddpm_basic import ddpm_simple
from models.unet import UNET
from models.utils import DDPM_Scheduler, set_seed
from timm.utils import ModelEmaV3
import numpy as np
import random
import math
import pdb
from tqdm import tqdm
from typing import List

def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()


num_time_steps=1000
ema_decay=0.9999
checkpoint_path="checkpoints/ddpm_flowers_epoch_1"
checkpoint = torch.load(checkpoint_path)
model = UNET(input_channels=3).cuda()
model.load_state_dict(checkpoint['weights'])
ema = ModelEmaV3(model, decay=ema_decay)
ema.load_state_dict(checkpoint['ema'])
scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
times = [0,15,50,100,200,300,400,550,700,999]
images = []

with torch.no_grad():
    model = ema.module.eval()
    for i in range(1):
        z = torch.randn(1, 3, 64, 64)
        for t in reversed(range(1, num_time_steps)):
            t = [t]
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
            z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(),t).cpu())
            if t[0] in times:
                images.append(z)
            e = torch.randn(1, 3, 64, 64)
            z = z + (e*torch.sqrt(scheduler.beta[t]))
        temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
        x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),[0]).cpu())

        images.append(x)
        x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
        x = x.numpy()
        plt.imshow(x)
        plt.show()
        display_reverse(images)
        images = []