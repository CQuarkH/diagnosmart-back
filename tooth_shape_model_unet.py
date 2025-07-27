import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# ----------------------
# U-Net multiclase en PyTorch
# ----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 64); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64,128); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128,256); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256,512); self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(512,1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024,512,2,2); self.dec4 = DoubleConv(1024,512)
        self.up3 = nn.ConvTranspose2d(512,256,2,2);  self.dec3 = DoubleConv(512,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,2);  self.dec2 = DoubleConv(256,128)
        self.up1 = nn.ConvTranspose2d(128, 64,2,2);  self.dec1 = DoubleConv(128, 64)
        # Output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x); p1=self.pool1(s1)
        s2 = self.enc2(p1); p2=self.pool2(s2)
        s3 = self.enc3(p2); p3=self.pool3(s3)
        s4 = self.enc4(p3); p4=self.pool4(s4)
        b  = self.bottleneck(p4)
        d4 = self.up4(b); d4=torch.cat([d4,s4],1); d4=self.dec4(d4)
        d3 = self.up3(d4); d3=torch.cat([d3,s3],1); d3=self.dec3(d3)
        d2 = self.up2(d3); d2=torch.cat([d2,s2],1); d2=self.dec2(d2)
        d1 = self.up1(d2); d1=torch.cat([d1,s1],1); d1=self.dec1(d1)
        return self.final_conv(d1)