# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/blocks.ipynb (unless otherwise specified).

__all__ = ['ConvLayer', 'DropConnect', 'SqueezeExpand', 'MBConvBlock', 'UnetBlock', 'BasicResBlock', 'res_blocks']

# Cell
# default_exp blocks
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

from fastcore.meta import delegates
from fastcore.basics import store_attr

# Cell
import sys
sys.path.append('..')
from .utils import all_equal

# Cell
class ConvLayer(nn.Sequential):
    " Construct a Sequence of Conv -> BN -> Act "
    @delegates(nn.Conv3d)
    def __init__(self,
                 in_c, # number of input channels
                 out_c, # number of output channels
                 ks=3, # kernel size (tuple or int)
                 stride=1, # kernel stride (tuple or int)
                 padding='auto', # padding during convolution, if auto padding is calcualted automatically
                 pad_value=0., # value to pad input with
                 norm=nn.BatchNorm3d, # type of batch nornalization
                 act=nn.ReLU, # activation function
                 transpose=False, # if transpose convolution should be constructed
                 **kwargs # further arguments for ConvLayer
                ):
        layers = OrderedDict([])

        # asymmetric padding
        if padding=='auto':
            if isinstance(ks, int): ks = (ks, )*3
            padding = [pad for _ks in ks for pad in self.calculate_padding(_ks)]
            if all_equal(padding): padding = padding[0]
            else: layers['pad'] = nn.ConstantPad3d(padding[::-1], value=pad_value)

        # Conv Layer
        Conv = nn.ConvTranspose3d if transpose else nn.Conv3d
        conv_layer = Conv(in_c, out_c, ks, stride=stride,
                          padding=0 if len(layers) == 1 else padding, **kwargs)

        layers['transpose_conv' if transpose else 'conv'] =  conv_layer

        if norm: layers['norm'] = norm(out_c)
        if act: layers['act'] = act()

        # create layers
        super().__init__(layers)

    def calculate_padding(self, ks):
        if ks % 2 == 0: return ks // 2, (ks-1) //2
        else: return ks //2, ks // 2

# Cell
class DropConnect(nn.Module):
    " Drops connections with probability p "
    def __init__(self,
                 p # percentage to drop
                ):
        assert 0 <= p <= 1, 'p must be in range of [0,1]'
        self.p = 1 - p # percentage to KEEP
        super().__init__()

    def forward(self, x):
        if not self.training: return x
        batch_size = x.size(0)

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = self.p + torch.rand([batch_size, 1, 1, 1, 1], dtype=x.dtype,
                                            device=x.device)
        return x / self.p

# Cell

class SqueezeExpand(nn.Module):
    "Squeeze Excitation Layer"
    @delegates(nn.Conv3d)
    def __init__(self,
                 in_c, # number of input channels
                 se_ratio, # squeeze-expand ratio
                 norm=nn.BatchNorm3d, # type of batch nornalization
                 act=nn.ReLU, # activation function
                 **kwargs # further arguments for ConvLayer
                ):
        super(SqueezeExpand, self).__init__()
        num_squeezed_channels = max(1, int(in_c * se_ratio))
        self.squeeze_expand = nn.Sequential(
            OrderedDict([
                ('pool', nn.AdaptiveAvgPool3d(1)),
                ('squeeze', ConvLayer(in_c=in_c, out_c=num_squeezed_channels, ks=1,
                            act=act, norm=None, **kwargs)),
                ('expand',  ConvLayer(in_c=num_squeezed_channels, out_c=in_c, ks=1,
                            act=None, norm=None,**kwargs)),
                ('sigmoid', nn.Sigmoid())])
        )
    def forward(self, x):
        return x * self.squeeze_expand(x)

# Cell

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self,
                 in_c, # number of input channels
                 out_c, # number of output channels
                 ks, # size of convolution kernel
                 stride, # stride of kernel
                 se_ratio, # squeeze-expand ratio
                 id_skip, # if skip connection shouldbe used
                 expand_ratio, # expansion ratio for inverted bottleneck
                 drop_connect_rate = 0.2, # percentage of dropped connections
                 act=nn.SiLU, # type of activation function
                 norm=nn.BatchNorm3d, # type of batch normalization
                 **kwargs # further arguments passed to `ConvLayerDynamicPadding`
                ):
        super(MBConvBlock, self).__init__()
        store_attr()

        # expansion phase (inverted bottleneck)
        n_intermed = in_c * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self.expand_conv = ConvLayer(in_c=in_c, out_c=n_intermed,
                                         ks = 1,norm=norm,
                                         act=act, **kwargs)

        # depthwise convolution phase, groups makes it depthwise
        self.depthwise_conv = ConvLayer(in_c=n_intermed, out_c=n_intermed,
                                        groups=n_intermed, ks=ks,
                                        stride=stride, norm=norm,
                                        act=act, **kwargs)

        # squeeze and excitation layer, if desired
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        if self.has_se:
            self.squeeze_expand = SqueezeExpand(in_c=n_intermed, se_ratio=se_ratio,
                                                act=act, norm=norm)

        # pointwise convolution phase
        self.project_conv = ConvLayer(in_c=n_intermed, out_c=out_c, ks=1,
                                      act = None, **kwargs)

        self.drop_conncet = DropConnect(drop_connect_rate)

    def forward(self, x):
        if self.id_skip: inputs = x # save input only if skip connection

        # expansion
        if self.expand_ratio != 1: x = self.expand_conv(x)

        # depthwise convolution
        x = self.depthwise_conv(x)

        # squeeze and excitation (self attention)
        if self.has_se:  x = self.squeeze_expand(x) * x

        # pointwise convolution
        x = self.project_conv(x)

        # skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.in_c == self.out_c:
            x = self.drop_conncet(x) + inputs  # skip connection
        return x

# Cell
class UnetBlock(nn.Module):
    " Create a U-Net Block "
    @delegates(nn.ConvTranspose3d)
    def __init__(self, up_c, s_c, ks=3, stride=2, act=nn.ReLU, norm=nn.BatchNorm3d, **kwargs):
        super(UnetBlock, self).__init__()

        self.up = ConvLayer(up_c, up_c//2, ks=ks, stride=stride, act=act, norm=None, transpose=True, **kwargs)
        self.bn = norm(s_c)

        in_c = up_c // 2 + s_c
        out_c = in_c // 2

        self.final_conv = nn.Sequential(
            act(),
            ConvLayer(in_c, out_c, act=act, norm=norm, **kwargs),
            ConvLayer(out_c, out_c, act=act, norm=norm, **kwargs) # ks = ?
        )

    def forward(self, up_in, s):
        s = self.bn(s)
        up_out = self.up(up_in)
        if s.shape[-3:] != up_out.shape[-3:]:
            up_out = F.interpolate(up_out, s.shape[-3:], mode='nearest')
        cat_x = torch.cat([up_out, s], dim=1)
        return self.final_conv(cat_x)

# Cell
class BasicResBlock(nn.Module):
    @delegates(ConvLayer)
    def __init__(self, in_c, out_c, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm=nn.BatchNorm3d, act=nn.ReLU, **kwargs):
        super(BasicResBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Sequential(OrderedDict([
            ('conv_layer_1', ConvLayer(in_c, out_c, ks=3, stride=stride, norm=norm, act=act, **kwargs)),
            ('conv_layer_2', ConvLayer(out_c, out_c, ks=3, stride=1, norm=norm, act=None, **kwargs))])
                                 )
        if stride != 1 or in_c != out_c:
            self.downsample = ConvLayer(in_c, out_c, ks=1, stride=stride, norm=norm, act=None, **kwargs)
        else: self.downsample = nn.Identity()

        self.final_act = act()

    def forward(self, x):
        x = self.conv(x) + self.downsample(x)
        return self.final_act(x)

# Cell
@delegates(BasicResBlock)
def res_blocks(in_c, out_c, stride, n_blocks, **kwargs):
    blocks = OrderedDict([('block_0', BasicResBlock(in_c, out_c, stride=stride, **kwargs))])
    if n_blocks == 1: return nn.Sequential(blocks)
    for i in range(n_blocks - 1):
        blocks[f'block_{i+1}'] = BasicResBlock(out_c, out_c, stride=1, **kwargs)
    return nn.Sequential(blocks)
