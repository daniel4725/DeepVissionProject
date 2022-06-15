import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class EnsembleModel(nn.Module):
    def __init__(self, models_lst, weights=None, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.models_lst = nn.ModuleList(models_lst)
        self.weights = weights
        if self.weights is None:
            self.weights = [1/len(models_lst)] * len(models_lst)
        self.to(device)

    def forward(self, x):
        out = self.weights[0] * self.models_lst[0](x)
        for model, w in zip(self.models_lst[1:], self.weights[1:]):
            out += model(x) * w
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        conv1 = nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3,
                          padding=1, bias=False)
        batch_norm1 = nn.BatchNorm3d(num_features=features)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3,
                          padding=1, bias=False)
        batch_norm2 = nn.BatchNorm3d(num_features=features)
        relu2 = nn.ReLU(inplace=True)
        return nn.Sequential(conv1, batch_norm1, relu1, conv2, batch_norm2, relu2)


if __name__ == "__main__":
    from Dataset import *
    from torch.utils.data import Dataset, DataLoader

    # Select GPU
    GPU_ID = '1'
    print('GPU USED: ' + GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if runes on one

    unet = UNet()
    unet.to(device)

    data_loaders = get_dataloaders(batch_size=1, data_type="Heart", base_data_dir='data',
                                   fold=0, num_workers=0, transform=None, load2ram=False)
    train_loader, valid_loader, test_loader = data_loaders

    for x, y in train_loader:
        x = x.float().to(device)
        out = unet(x)