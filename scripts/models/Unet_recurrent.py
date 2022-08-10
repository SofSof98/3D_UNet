import torch.nn as nn
from models.Unet_utils import *
from utils.init_weights import init_weights

class Unet_3D(nn.Module):

   
    def __init__(self, input_shape, output_shape, opt, num_classes = 2):
        super(Unet_3D, self).__init__()
        
        self.in_channels = opt.n_channels
        self.is_batchnorm = opt.batchnorm
        self.feature_scale = opt.filter_scale
        self.norm_type = opt.norm_type
    
        batchNormObject = lambda n_filters: nn.GroupNorm(opt.n_groups, n_filters) if self.norm_type != 'batch' else nn.BatchNorm3d(n_filters) 
 
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # downsampling
        self.conv_block_1 = RCNN_block(self.in_channels, filters[0] // 2, filters[0], batchNormObject ,time=2,dr =opt.dr) 
        self.conv_block_2 = RCNN_block(filters[0], filters[0], filters[1], batchNormObject ,time=2,dr =opt.dr)
        self.conv_block_3 = RCNN_block(filters[1], filters[1], filters[2],batchNormObject ,time=2,dr =opt.dr)
        

        
        self.center_block = RCNN_block(filters[2], filters[2], filters[3],batchNormObject ,time=2,dr =opt.dr)

        # upsampling
        self.up_concat3 = Up_Conv_3D(filters[3], filters[3], filters[2], block =RCNN_block, batchNormObject=batchNormObject ,time=2,dr =opt.dr)
        self.up_concat2 = Up_Conv_3D(filters[2], filters[2], filters[1], block =RCNN_block, batchNormObject=batchNormObject ,time=2,dr =opt.dr)
        self.up_concat1 = Up_Conv_3D(filters[1], filters[1], filters[0], block =RCNN_block, batchNormObject=batchNormObject ,time=2,dr =opt.dr)

        # final conv (without any concat)
        self.final = Output_block_binary(filters[0], 1)
        #self.final = Output_block(filters[0], num_classes, self.is_batchnorm)

         # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')



    def forward(self, inputs):
        conv1 = self.conv_block_1(inputs)
        max_pool1 = self.maxpool(conv1)

        conv2 = self.conv_block_2(max_pool1)
        max_pool2 = self.maxpool(conv2)

        conv3 = self.conv_block_3(max_pool2)
        max_pool3 = self.maxpool(conv3)

    

        center_block = self.center_block(max_pool3)
        up3 = self.up_concat3(conv3, center_block)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        outputs = self.final(up1)

        return outputs

