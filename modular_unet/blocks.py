# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/blocks.ipynb (unless otherwise specified).

__all__ = ['ConvLayer', 'DropConnect', 'SqueezeExpand', 'MBConvBlock', 'SpatialAttention', 'SpatialAttentionDualInput',
           'UnetBlock', 'ResBlock', 'DoubleConv', 'DeepSupervision', 'res_blocks']

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
from .utils import all_equal, test_forward

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
class SpatialAttention(nn.Module):
    "Apply attention gate to input in U-Net Block. Adapted from arxiv.org/abs/1804.03999"
    def __init__(self,
                 in_c, # number of input channels
                 ks = 7,
                 max_pool=True,
                 mean_pool=True,
                 xtra_conv=False,
                 **kwargs # further arguments for ConvLayer
                ):
        super(SpatialAttention, self).__init__()
        store_attr()

        if self.xtra_conv:
            self.xtra_conv_layer = ConvLayer(in_c, 1, ks=ks, **kwargs)

        n_final = max_pool + mean_pool + xtra_conv
        assert n_final > 0, 'No pooling layers in `SpatialAttention`-block'
        self.out_conv = ConvLayer(n_final, 1, ks=ks, act=nn.Sigmoid, norm=None, **kwargs)


    def forward(self, x):
        compressed = list()
        if self.max_pool: compressed.append(x.max(1)[0].unsqueeze(1))
        if self.mean_pool: compressed.append(x.mean(1).unsqueeze(1))
        if self.xtra_conv: compressed.append(self.xtra_conv_layer(x))
        return self.out_conv(torch.cat(compressed, 1)) * x

# Cell
class SpatialAttentionDualInput(nn.Module):
    "Apply attention gate to input in U-Net Block. Adapted from arxiv.org/abs/1804.03999"
    def __init__(self,
                 in_c, # number of input channels
                 s_c, # number of gated channels
                 **kwargs # further arguments for ConvLayer
                ):
        super(SpatialAttentionDualInput, self).__init__()
        self.conv_u = ConvLayer(in_c, s_c, ks=1, stride=1, act=None, norm=None)
        self.conv_s = ConvLayer(s_c, s_c, ks=2, stride=2,  act=None, norm=None, bias = False)
        self.conv_attn = nn.Sequential(
            nn.ReLU(),
            ConvLayer(s_c, 1, ks=1,  act=nn.Sigmoid, stride=1, **kwargs),
        )

    def forward(self, up_in, s):
        x = self.conv_u(up_in)
        s = F.interpolate(self.conv_s(s), size=x.shape[2:], mode='trilinear', align_corners=False)
        attn_gate = F.interpolate(self.conv_attn(x + s), size=up_in.shape[2:], mode='trilinear', align_corners=False)
        return up_in * attn_gate

# Cell
class UnetBlock(nn.Module):
    " Create a U-Net Block "
    @delegates(nn.ConvTranspose3d)
    def __init__(self,
                 in_c, # number of input channels
                 s_c, # number of gated channels
                 ks=3, # kernel size for upsampling layer
                 stride=2, # stride for upsampling layer
                 norm=nn.BatchNorm3d, # type of batch nornalization
                 act=nn.ReLU, # activation function
                 spatial_attention=False, # use spatial attention on input
                 **kwargs # further arguments for ConvLayer
                ):
        super(UnetBlock, self).__init__()
        up_c = in_c # in_c is used with mist Modules, but up_c would be a more suitable name for this block

        self.up = ConvLayer(up_c, up_c//2, ks=ks, stride=stride, act=act, norm=None, transpose=True, **kwargs)
        self.bn = norm(s_c)
        if spatial_attention: self.sa = SpatialAttentionDualInput(up_c, s_c)

        in_c = up_c // 2 + s_c
        out_c = in_c // 2

        self.final_conv = nn.Sequential(
            act(),
            ConvLayer(in_c, out_c, act=act, norm=norm, **kwargs),
            ConvLayer(out_c, out_c, act=act, norm=norm, **kwargs) # ks = ?
        )

    def forward(self, up_in, s):
        s = self.bn(s)
        if hasattr(self, 'sa'): up_in = self.sa(up_in, s)
        up_out = self.up(up_in)
        if s.shape[-3:] != up_out.shape[-3:]:
            up_out = F.interpolate(up_out, s.shape[-3:], mode='nearest')
        cat_x = torch.cat([up_out, s], dim=1)
        return self.final_conv(cat_x)

# Cell
class ResBlock(nn.Module):
    " ResBlock like implementation for 3D"

    @delegates(ConvLayer.__init__)
    def __init__(self, in_c, out_c, ks=3, stride=1, padding='auto', bottleneck=True,
                 base_width=64, groups = 1, norm=nn.BatchNorm3d, act=nn.ReLU, **kwargs):
        super(ResBlock, self).__init__()

        width = int(out_c * (base_width / 64.)) * groups
        if not bottleneck:
            assert groups == 1 and base_width == 64, \
            'Expected base_width to be 64 and groups to be 1 with expansion 1'

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv = OrderedDict([
            ('conv_layer_1', ConvLayer(in_c, width, ks=1 if bottleneck else ks,
                                       stride=1 if bottleneck else stride, padding=padding,
                                       norm=norm, act=act, **kwargs)),
            ('conv_layer_2', ConvLayer(width, width, ks=ks, stride=stride if bottleneck else 1,
                                       padding=padding, norm=norm, act=None, **kwargs))]
            )

        if bottleneck:
            conv['conv_layer_3'] = ConvLayer(width, out_c, ks=1, stride=1,
                                             padding=padding, norm=norm, act=None, **kwargs)

        self.conv = nn.Sequential(conv)

        if stride != 1 or in_c != out_c:
            self.downsample = ConvLayer(in_c, out_c, ks=1, stride=stride, norm=norm, act=None, **kwargs)
        else: self.downsample = nn.Identity()
        self.final_act = act()

    def forward(self, x):
        x = self.conv(x) + self.downsample(x)
        return self.final_act(x)

# Cell
class DoubleConv(nn.Module):
    @delegates(ConvLayer)
    def __init__(self, in_c, **kwargs):
        super(DoubleConv, self).__init__()
        self.conv1 = ConvLayer(in_c, in_c*2)
        self.conv2 = ConvLayer(in_c*2, in_c)

    def forward(self, x):
        return self.conv2(self.conv1(x))

# Cell
class DeepSupervision(nn.Module):
    @delegates(ConvLayer)
    def __init__(self, in_c, out_c, ks=1, act=None, norm=None, **kwargs):
        super(DeepSupervision, self).__init__()
        assert out_c > 1, f'Expected `out_c` to be at least 2 but got {out_c}'
        self.conv = nn.Sequential(
            ConvLayer(in_c, out_c, ks=ks, act=act, norm=norm),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.conv(x)


# Cell
# deprecated
@delegates(ResBlock)
def res_blocks(in_c, out_c, stride, n_blocks, **kwargs):
    blocks = OrderedDict([('block_0', ResBlock(in_c, out_c, stride=stride, **kwargs))])
    if n_blocks == 1: return nn.Sequential(blocks)
    for i in range(n_blocks - 1):
        blocks[f'block_{i+1}'] = BasicResBlock(out_c, out_c, stride=1, **kwargs)
    return nn.Sequential(blocks)