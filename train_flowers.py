import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
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
import pathlib

IMAGE_SIZE = 60
AUGMENTATION_FACTOR = 1000
batch_size = 64
num_time_steps = 1000
num_epochs = 30
seed = -1
ema_decay = 0.9999
lr = 2e-5
checkpoint_path = None

set_seed(random.randint(0, 2 ** 32 - 1)) if seed == -1 else set_seed(seed)

data_folder = pathlib.Path(
    r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data\flower\flower_data\flower_data\flower_subset")


def scale_to_neg_one_to_one(x):
    return x * 2 - 1


# Transformations for the dataset
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # Converts image to [0,1]
])

# Replace 'data' with your dataset directory
original_dataset = datasets.ImageFolder(root=str(data_folder), transform=transform)
augmented_datasets = [original_dataset] * AUGMENTATION_FACTOR
train_dataset = ConcatDataset(augmented_datasets)

# sub_dataset = Subset(train_dataset, list(range(1024)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
model = UNET(input_channels=3).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
ema = ModelEmaV3(model, decay=ema_decay)

if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['weights'])
    ema.load_state_dict(checkpoint['ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# We will use Mean Squared Error loss to compare predicted noise against actual noise
criterion = nn.MSELoss(reduction='mean')
losses = []
min_avg_loss = 1e9

for i in range(num_epochs):
    total_loss = 0
    for bidx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")):
        x = x.cuda()
        x = F.pad(x, (2, 2, 2, 2))
        t = torch.randint(0, num_time_steps, (batch_size,))
        e = torch.randn_like(x, requires_grad=False)
        a = scheduler.alpha[t].view(batch_size, 1, 1, 1).cuda()
        x = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
        output = model(x, t)
        optimizer.zero_grad()
        loss = criterion(output, e)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        ema.update(model)
    avg_loss = total_loss / (60000 / batch_size)
    print(f'Epoch {i + 1} | Loss {avg_loss:.5f}')
    losses.append(avg_loss)

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }

    if avg_loss <= min_avg_loss:
        min_avg_loss = avg_loss
        torch.save(checkpoint, f'checkpoints/ddpm_flowers_epoch_{i + 1}')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Loss plot across epochs")
plt.show()
