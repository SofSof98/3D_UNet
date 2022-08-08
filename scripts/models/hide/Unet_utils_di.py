import torch
import torch.nn as nn

class Conv_3D_block_di(nn.Module):
    def __init__(self, in_size, mid_size,out_size,is_batchnorm,kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)):
        super(Conv_3D_block, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size,mid_size, kernel_size, stride, padding),
                                       nn.BatchNorm3d(mid_size),
                                       nn.ReLU(inplace=False),)
            self.conv2 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride,'same', dilation=2),
                                       nn.BatchNorm3d(mid_size),
                                       nn.ReLU(inplace=False),)
            self.conv3 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride, 'same', dilation=4),
                                       nn.BatchNorm3d(mid_size),
                                       nn.ReLU(inplace=False),)
            self.conv4 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride, 'same', dilation=8),
                                       nn.BatchNorm3d(mid_size),
                                       nn.ReLU(inplace=False),)
            self.final = nn.Sequential(nn.Conv3d(mid_size*4, out_size, (1,1,1), stride),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=False),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride, padding),
                                       nn.ReLU(inplace=False),)
            self.conv2 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride, padding, dilation=2),
                                       nn.ReLU(inplace=False),)
            self.conv3 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride, padding, dilation=4),
                                       nn.ReLU(inplace=False),)
            self.conv4 = nn.Sequential(nn.Conv3d(in_size, mid_size, kernel_size, stride, padding, dilation=8),
                                       nn.ReLU(inplace=False),)
            self.final = nn.Sequential(nn.Conv3d(mid_size*4, out_size, (1,1,1), stride),
                                       nn.ReLU(inplace=False),)
        
        
    def forward(self, inputs):
        outputs_1 = self.conv1(inputs)
        #print(outputs_1.shape)
        outputs_2 = self.conv2(inputs)
        #print(outputs_2.shape)
        outputs_3 = self.conv3(inputs)
        #print(outputs_3.shape)
        outputs_4 = self.conv4(inputs)
        #print(outputs_4.shape)
        outputs = torch.cat([outputs_1, outputs_2, outputs_3, outputs_4], 1)
        outputs = self.final(outputs)
        return outputs

class Up_Conv_3D_block_di(nn.Module):
    def __init__(self, in_size, mid_size, out_size, is_batchnorm=True):
        super(Up_Conv_3D_block, self).__init__()
        
        
        self.up = nn.ConvTranspose3d(in_size, mid_size, kernel_size=(2,2,2), stride=(2,2,2))
        self.up_conv = Conv_3D_block(mid_size + out_size, out_size, out_size, is_batchnorm)
        
    def forward(self,skip, inputs):
        out_up = self.up(inputs)
        merged_skip = torch.cat([skip, out_up], 1)
        outputs = self.up_conv(merged_skip)
        return outputs

class Output_block_binary(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(Output_block_binary, self).__init__()
        
        if is_batchnorm:
            self.conv =  nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0),
                                       nn.BatchNorm3d(out_size),
                                       nn.Sigmoid(),)

        else:
            self.conv =  nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0),
                                       nn.Sigmoid(),)        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class Output_block_multi(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(Output_block_multi, self).__init__()
        
        if is_batchnorm:
            self.conv =  nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0),
                                       nn.BatchNorm3d(out_size),
                                       nn.Softmax(dim=1),)

        else:
            self.conv =  nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0),
                                       nn.Softmax(dim=1),)        
    def forward(self,inputs):
        outputs = self.conv(inputs)
        return outputs


class Single_Conv3D(nn.Module):
    def __init__(self, in_size, out_size, is_first_block,kernel_size=(3,3,3),padding=(1,1,1), stride=(1,1,1)):
        super(Single_Conv3D, self).__init__()

        if is_first_block:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=False),)
            
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, stride, padding),
                                       nn.BatchNorm3d(out_size),)
            
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class Single_Conv3D_Res(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3,3,3),padding=(1,1,1), stride=(1,1,1)):
        super(Single_Conv3D_Res, self).__init__()

       
        self.conv1 = nn.Sequential(nn.BatchNorm3d(in_size),
                                       nn.ReLU(inplace=False),
                                       nn.Conv3d(in_size, out_size, kernel_size, stride, padding),)
         
                
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class Classic_Residual_Block(nn.Module):
    def __init__(self, in_size, mid_size, out_size, is_first_block):
        super(Classic_Residual_Block, self).__init__()

        self.res1 = Single_Conv3D(in_size, mid_size)
        self.res2 = Single_Conv3D(mid_size, out_size, is_first_block= False)
        self.activation = nn.ReLU(inplace=False)
        self.short_conv = nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0)

    def forward(self, inputs):
        residual_1 = self.res1(inputs)
        residual_2 = self.res2(residual_1)
        shortcut = self.short_conv(inputs)
        outputs= torch.add(shortcut, residual_2)
        outputs = self.activation(outputs)
        return outputs

class Residual_Block(nn.Module):
    
    def __init__(self, in_size, mid_size, out_size):
        super(Residual_Block, self).__init__()

        self.res1 = Single_Conv3D_Res(in_size, mid_size)
        self.res2 = Single_Conv3D_Res(mid_size, out_size)
        self.short_conv = nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0)

    def forward(self, inputs):
        residual_1 = self.res1(inputs)
        residual_2 = self.res2(residual_1)
        shortcut = self.short_conv(inputs)
        outputs= torch.add(shortcut, residual_2)
        return outputs

        
        
class Up_Conv_3D_Res(nn.Module):
    def __init__(self, in_size,mid_size, out_size):
        super(Up_Conv_3D_Res, self).__init__()
        
        
        self.up = nn.ConvTranspose3d(in_size, mid_size, kernel_size=(2,2,2), stride=(2,2,2))
        self.up_conv = Residual_Block(mid_size + out_size, out_size, out_size)
        
    def forward(self,skip, inputs):
        out_up = self.up(inputs)
        merged_skip = torch.cat([skip, out_up], 1)
        outputs = self.up_conv(merged_skip)
        return outputs


class First_Res_Block(nn.Module):
    def __init__(self, in_size, mid_size, out_size,kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)):
        super(First_Res_Block, self).__init__()
        self.res =  Single_Conv3D(in_size, mid_size, is_first_block= False)
        self.conv = nn.Conv3d(mid_size, out_size, kernel_size, stride, padding)
        self.short_conv = nn.Conv3d(in_size, out_size, kernel_size=1, stride =(1,1,1), padding = 0)
    
    def forward(self, inputs):
        residual = self.res(inputs)
        residual = self.conv(residual)
        shortcut = self.short_conv(inputs)
        outputs= torch.add(shortcut, residual)
        return outputs


## Attention unet

class Attention_block(nn.Module):
    def __init__(self, in_size, mid_size, gating_size,kernel_size=(2,2,2), stride=(2,2,2)):
        super(Attention_block, self).__init__()
        '''
        x is the skip connection
        g is the gating signal
        in_size is the n.filters of the skip
        mid_size is the n.filters of the skip
        gating_size is the n.filters of the gating
        '''
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.conv_theta_x = nn.Conv3d(in_size, mid_size, kernel_size, stride, padding=0, bias=False)
        self.conv_phi_g = nn.Conv3d(gating_size, mid_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_psi = nn.Conv3d(mid_size,1, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=False)
        self.sigm = nn.Sigmoid()
        # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
        self.W = nn.Sequential(
            nn.Conv3d(in_size,in_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(in_size),
        )

        
    def forward(self,x,g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.conv_theta_x(x)
        theta_x_size = theta_x.size()
        phi_g = self.conv_phi_g(g)
        if phi_g.size(2) != theta_x_size[2]:
            phi_g = nn.functional.interpolate(phi_g, size=theta_x_size[2:], mode='trilinear')
        f = self.act(theta_x + phi_g)
        psi = self.conv_psi(f)
        # attention scores
        sigma_psi = self.sigm(psi)

        # upsample attention map
        sigma_psi = nn.functional.interpolate(sigma_psi, size=input_size[2:], mode='trilinear')
        # expand attention map to n.filters of x and multiply by x
        out = sigma_psi.expand_as(x)*x
        W_out = self.W(out)

        return W_out, sigma_psi

        
            
class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=False),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=False),
                                       )
            
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
        
class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gating_size, mid_size, is_conv=False):
        super(MultiAttentionBlock, self).__init__()
        self.is_conv = is_conv
        self.gate_block_1 = Attention_block(in_size=in_size, gating_size=gating_size,
                                                 mid_size=mid_size)
        
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=False),
                                           )

    def forward(self, inputs, gating_signal):
        gate_1, attention_1 = self.gate_block_1(inputs, gating_signal)

        if self.is_conv:
            gate_1 = self.combine_gates(gate_1)
        return gate_1, attention_1


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'),)

    def forward(self, input):
        return self.dsv(input)
        
        
## Resnet Unet blocks

'''
class Resnet_block(nn.Module):
     def __init__(self, in_size, out_size,filter_scale=1,kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1), stride = ):
        super(Resnet_block, self).__init__()

        self.res = Residual_Block(in_size, out_size)

        def forward(self, inputs, repetitions)
     
'''
