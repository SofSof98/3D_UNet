"""
Copyright Dejan Kostyszyn 2019
"""

import torch, utils
import torch.nn as nn
from utils.init_weights import init_weights

class Unet_3D(nn.Module):

    def __init__(self, input_shape, output_shape, opt, num_classes = 2):
        super(Unet_3D, self).__init__()

        self.opt = opt
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.groups = opt.n_groups
        
        self.enc1 = Encoder(2, 32, 64, self.groups)
        self.enc2 = Encoder(64, 64, 128, self.groups)
        self.enc3 = Encoder(128, 128, 256, self.groups)

        self.bottom = Bottom(256, 512, self.groups)

        self.dec3 = Decoder(768, 256, 256, self.groups)
        self.dec2 = Decoder(384, 128, 128, self.groups)

        self.dec1 = Final(192, 64, num_classes, self.groups)

                # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')



    def forward(self, x):
        # Encoding.
        x1 = self.enc1(x)
        x = self.maxpool(x1)
        x2 = self.enc2(x)
        x = self.maxpool(x2)
        x3 = self.enc3(x)
        x = self.maxpool(x3)

        # Bottom.
        x = self.bottom(x)
        
        # Decoding.
        x = torch.cat((x3, x), 1)
        x = self.dec3(x)
        x = torch.cat((x2, x), 1)
        x = self.dec2(x)
        x = torch.cat((x1, x), 1)
        x = self.dec1(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, groups):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,groups):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,middle_channels),
            nn.ReLU(),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,middle_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.decoder(x)

class Bottom(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(Bottom, self).__init__()
        self.bottom = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.bottom(x)

class Final(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, groups):
        super(Final, self).__init__()
        self.final = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups,mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, padding=0),
            #nn.GroupNorm(groups,out_channels),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.final(x)
