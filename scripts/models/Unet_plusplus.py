import torch.nn as nn
from models.Unet_utils import *


class Unet_3D(nn.Module):

    def __init__(elf, input_shape, output_shape, opt, num_classes = 2 ,is_deconv = True, deep_supervision = True):
        super(Unet_3D, self).__init__()

        self.in_channels = opt.n_channels
        self.is_batchnorm = opt.batchnorm
        self.feature_scale = opt.filter_scale
        self.norm_type = opt.norm_type
        self.deconv = is_deconv
        self.deep_supervision = deep_supervision
        
        batchNormObject = lambda n_filters: nn.GroupNorm(opt.n_groups, n_filters) if self.norm_type != 'batch' else nn.BatchNorm3d(n_filters)
    
 

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        
        
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # downsampling
        self.conv_block_1 = Conv_3D_block(self.in_channels, filters[0], filters[0], self.is_batchnorm, batchNormObject, opr.dr)
        self.conv_block_2 = Conv_3D_block(filters[0], filters[1], filters[1], self.is_batchnorm, batchNormObject, opr.dr)
        self.conv_block_3 = Conv_3D_block(filters[1], filters[2], filters[2], self.is_batchnorm, batchNormObject, opr.dr)
        self.conv_block_4 = Conv_3D_block(filters[2], filters[3], filters[3], self.is_batchnorm, batchNormObject, opr.dr)
        self.center_block = Conv_3D_block(filters[3], filters[4], filters[4], self.is_batchnorm, batchNormObject, opr.dr)

        # upsampling
        self.up_concat01 = Nested_block(filters[1], filters[0], filters[0],self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat02 = Nested_block(filters[1], filters[0]*2, filters[0],self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat03 = Nested_block(filters[1], filters[0]*3, filters[0], self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat11 = Nested_block(filters[2],filters[1], filters[1],self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat12 = Nested_block(filters[2], filters[1]*2, filters[1], self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat21 = Nested_block(filters[3] , filters[2], filters[2], self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
	
        self.up_concat22 = Nested_block(filters[3],filters[2]*2, filters[2],self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat13 = Nested_block(filters[2], filters[1]*3, filters[1], self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat31 = Nested_block(filters[4] , filters[3], filters[3], self.deconv, self.is_batchnorm, batchNormObject, opr.dr)
        self.up_concat04 = Nested_block(filters[1], filters[0]*4, filters[0], self.deconv, self.is_batchnorm, batchNormObject, opr.dr)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)
        self.final2 = nn.Conv3d(filters[0], 1, 1)
        self.final3= nn.Conv3d(filters[0], 1, 1)
        self.final4= nn.Conv3d(filters[0], 1, 1)
        #self.final = Output_block(filters[0], num_classes, self.is_batchnorm)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        
        x_00 = self.conv_block_1(inputs) # x_00
        max_pool1 = self.maxpool(x_00)

        x_10 = self.conv_block_2(max_pool1) # x_10
        max_pool2 = self.maxpool(x_10)

        x_20 = self.conv_block_3(max_pool2) # x_20
        max_pool3 = self.maxpool(x_20)
        
        x_30 = self.conv_block_4(max_pool3) # x_30
        max_pool4 = self.maxpool(x_30)

        x_40 = self.center_block(max_pool4) # x_40


        x_01 = self.up_concat01(x_10,x_00)
        x_11 = self.up_concat11(x_20,x_10)
        x_21 = self.up_concat21(x_30, x_20)
        x_31 = self.up_concat31(x_40,x_30)

        x_02 = self.up_concat02(x_11, x_01, x_00)
        x_12 = self.up_concat12(x_21, x_11, x_10)
        x_22 = self.up_concat22(x_31, x_21, x_20)
        
        x_03 = self.up_concat03(x_12, x_02, x_01, x_00)
        x_13 = self.up_concat13(x_22, x_12, x_11, x_10)
        x_04 = self.up_concat04(x_13, x_03, x_02, x_01, x_00)

        if self.deep_supervision:
            output1 = self.final(x_01)
            output2 = self.final2(x_02)
            output3 = self.final3(x_03)
            output4 = self.final4(x_04)
            outputs = (output1 + output2 + output3 + output4)/4
            

        else:
            outputs =self.final(x_01)

        return self.sigmoid(outputs)




class Nested_block(nn.Module):

    def __init__(self, in_size, mid_size, out_size, deconv, is_batchnorm=True, batchNormObject = nn.BatchNorm3d, dr =0.0):
        super(Nested_block, self).__init__()

        self.conv_up = Conv_3D_block(out_size + mid_size, out_size, out_size, is_batchnorm, batchNormObject, dr)
        if deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(2,2,2), stride=(2,2,2))
        else:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, inputs, *input):
        inputs = self.up(inputs)
        for i in range(len(input)):
            inputs = torch.cat([input[i],inputs],1)
        outputs = self.conv_up(inputs)
        return outputs
        
                    
