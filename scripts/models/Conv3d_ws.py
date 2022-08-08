import torch
import torch.nn as nn
from torch.nn import functional as F



def weight_standardization(weight: torch.Tensor, eps: float):
    r"""
    ## Weight Standardization
    $$\hat{W}_{i,j} = \frac{W_{i,j} - \mu_{W_{i,\cdot}}} {\sigma_{W_{i,\cdot}}}$$
    where,
    \begin{align}
    W &\in \mathbb{R}^{O \times I} \\
    \mu_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W_{i,j} \\
    \sigma_{W_{i,\cdot}} &= \sqrt{\frac{1}{I} \sum_{j=1}^I W^2_{i,j} - \mu^2_{W_{i,\cdot}} + \epsilon} \\
    \end{align}
    for a 2D-convolution layer $O$ is the number of output channels ($O = C_{out}$)
    and $I$ is the number of input channels times the kernel size ($I = C_{in} \times k_H \times k_W$)
    """

    # Get $C_{out}$, $C_{in}$ and kernel shape
    c_out, c_in, *kernel_shape = weight.shape
    # Reshape $W$ to $O \times I$
    weight = weight.view(c_out, -1)
    # Calculate
    #
    # \begin{align}
    # \mu_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W_{i,j} \\
    # \sigma^2_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W^2_{i,j} - \mu^2_{W_{i,\cdot}}
    # \end{align}
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    # Normalize
    # $$\hat{W}_{i,j} = \frac{W_{i,j} - \mu_{W_{i,\cdot}}} {\sigma_{W_{i,\cdot}}}$$
    weight = (weight - mean) / (torch.sqrt(var + eps))
    # Change back to original shape and return
    return weight.view(c_out, c_in, *kernel_shape)



class Conv3d(nn.Conv3d):
    """
    ## 2D Convolution Layer
    This extends the standard 2D Convolution layer and standardize the weights before the convolution step.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 eps: float = 1e-5):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias,
                                     padding_mode=padding_mode)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.conv3d(x, weight_standardization(self.weight, self.eps), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def _test():
    """
    A simple test to verify the tensor sizes
    """
    conv3d = Conv3d(10, 20, 5)
    from labml.logger import inspect
    inspect(conv3d.weight)
    import torch
    inspect(conv3d(torch.zeros(10, 10, 100, 100)))


if __name__ == '__main__':
    _test()
