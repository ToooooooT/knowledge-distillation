import argparse
import os
import random
from tqdm import tqdm

from models.SimpleSegViT.SimpleSegViT import SimpleSegViT
from torchsummary import summary
import torchvision

from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import mmcv
from mmengine import Config
from mmengine.runner import Runner
from mmseg.apis import inference_model
from mmseg.models import build_segmentor

from losses.atm_loss import ATMLoss

from dataset.ade20k_dataset import ade20k_dataset

from decode_heads import atm_head

from utils.util import intersectionAndUnionGPU


cfg = Config.fromfile('./configs/segvit/segvit_vit-l_jax_640x640_160k_ade20k.py')

test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608))
teacher = build_segmentor(cfg=cfg.model)
checkpoint = torch.load('./pretrained/vit_large_ade.pth')
del checkpoint['state_dict']['decode_head.loss_decode.criterion.empty_weight']
teacher.load_state_dict(checkpoint['state_dict'])
teacher = teacher.to('cuda')

crop_size = (512, 512)

train_data = ade20k_dataset('./configs/segvit/segvit_vit-l_jax_640x640_160k_ade20k.py', mode='train', crop_size=(512, 512), isTeach=True)
train_loader = DataLoader(train_data,
                        num_workers=32,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=False)

teacher = teacher.eval()
transform = transforms.Compose([
    transforms.Resize(crop_size, antialias=True)
])

for img, label, path in tqdm(train_loader):
    dest_path = path[0].split('/')
    dest_path[3] = 'annotations'
    dest_path[4] = 'teacher_label'
    dest_path[5] = dest_path[5][:-3] + 'pth'
    dest_path = '/'.join(dest_path)
    img = img.to('cuda')
    with torch.no_grad():
        teacher_pred = teacher(img)
    teacher_pred = teacher_pred.cpu()
    teacher_pred = transform(teacher_pred)

    B, C, H, W = img.shape
    teacher_pred = torch.cat([torch.zeros((B, 1, H, W)).to('cuda').to(torch.float), teacher_pred], dim=1)
    # Create a mask where the label is 0, shape will be [1, H, W]
    mask = (label == 0)
    teacher_pred[:, 0, :, :] = (teacher_pred[:, 0, :, :].to(torch.int) | mask.squeeze().to(torch.int)).to(torch.float)

    torch.save(teacher_pred[0], dest_path)



