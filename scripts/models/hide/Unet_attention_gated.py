import torch.nn as nn
from models.Unet_utils import *


class Unet_3D(nn.Module):

    def __init__(self, input_shape, output_shape, feature_scale=1, num_classes = 2, in_channels=1, is_batchnorm=True):
        super(Unet_3D, self).__init__()
        
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # downsampling
        self.conv_block_1 = Conv_3D_block(self.in_channels, filters[0], filters[0], self.is_batchnorm)
        self.conv_block_2 = Conv_3D_block(filters[0], filters[1], filters[1], self.is_batchnorm)
        self.conv_block_3 = Conv_3D_block(filters[1], filters[2], filters[2], self.is_batchnorm)
        self.conv_block_4 = Conv_3D_block(filters[2], filters[3], filters[3], self.is_batchnorm)
        
        self.gating = UnetGridGatingSignal3(filters[4], filters[3], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        
        self.center_block = Conv_3D_block(filters[3], filters[4], filters[4],self.is_batchnorm)

        # upsampling
        self.up_concat4 = Up_Conv_3D_block(filters[4], filters[3], filters[3], self.is_batchnorm)
        self.up_concat3 = Up_Conv_3D_block(filters[3], filters[2], filters[2], self.is_batchnorm)
        self.up_concat2 = Up_Conv_3D_block(filters[2], filters[1], filters[1], self.is_batchnorm)
        self.up_concat1 = Up_Conv_3D_block(filters[1], filters[0], filters[0], self.is_batchnorm)

        # attention blocks

        self.att2 = Attention_block(in_size=filters[1], gating_size=filters[3],mid_size=filters[1])
        self.att3 = Attention_block(in_size=filters[2], gating_size=filters[3],mid_size=filters[2])
        self.att4 = Attention_block(in_size=filters[3], gating_size=filters[3],mid_size=filters[3])
        
        
        # final conv (without any concat)
        self.final = Output_block_binary(filters[0], 1, self.is_batchnorm)
        #self.final = Output_block(filters[0], num_classes, self.is_batchnorm)
        


    def forward(self, inputs):
        
        # encoder
        conv1 = self.conv_block_1(inputs)
        max_pool1 = self.maxpool(conv1)

        conv2 = self.conv_block_2(max_pool1)
        max_pool2 = self.maxpool(conv2)

        conv3 = self.conv_block_3(max_pool2)
        max_pool3 = self.maxpool(conv3)

        conv4 = self.conv_block_4(max_pool3)
        max_pool4 = self.maxpool(conv4)


        center_block = self.center_block(max_pool4)
        gating = self.gating(center_block)

        # attention mechanism
        g_4, att4 = self.att4(conv4, gating)
        g_3, att3 = self.att3(conv3, gating)
        g_2, att2 = self.att2(conv2, gating)

        # decoder
        up4 = self.up_concat4(g_4, center_block)
        up3 = self.up_concat3(g_3, up4)
        up2 = self.up_concat2(g_2, up3)
        up1 = self.up_concat1(conv1, up2)

        outputs = self.final(up1)

        return outputs

