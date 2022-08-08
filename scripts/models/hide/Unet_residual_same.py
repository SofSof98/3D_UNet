
# res unet as in "road extraction by deep residual Unet"
import torch.nn as nn
from models.Unet_utils import *


class Unet_3D(nn.Module):

    def __init__(self, input_shape, feature_scale=1, num_classes = 2, in_channels=3, is_batchnorm=True):
        super(unet_3D, self).__init__()
        
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # downsampling
        self.conv_block_1 = First_Res_Block(self.in_channels, filters[0], filter_scale = 1, self.is_batchnorm)
        self.conv_block_2 = Residual_Block(filters[0], filters[1], filter_scale = 1, self.is_batchnorm)
        self.conv_block_3 = Residual_Block(filters[1], filters[2], filter_scale = 1, self.is_batchnorm)
        

        
        self.center_block = Residual_Block(filters[2], filters[3], filter_scale = 1, self.is_batchnorm)

        # upsampling
        self.up_concat3 = Up_Conv_3D_Res(filters[3], filters[2], filter_scale = 1, self.is_batchnorm)
        self.up_concat2 = Up_Conv_3D_Res(filters[2], filters[1], filter_scale = 1, self.is_batchnorm)
        self.up_concat1 = Up_Conv_3D_Res(filters[1], filters[0], filter_scale = 1, self.is_batchnorm)

        # final conv (without any concat)
        self.final = Output_block_binary(filters[0], 1, self.is_batchnorm)
        #self.final = Output_block(filters[0], num_classes, self.is_batchnorm)
        


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        max_pool1 = self.maxpool(conv1)

        conv2 = self.conv2(max_pool1)
        max_pool2 = self.maxpool(conv2)

        conv3 = self.conv3(max_pool2)
        max_pool3 = self.maxpool(conv3)

    

        center_block = self.center_block(max_pool3)
        up3 = self.up_concat3(conv3, center_block)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        outputs = self.final(up1)

        return outputs


