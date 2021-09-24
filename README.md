# Modular 3D U-Net
> A modular U-net that allows the quick experimentation with different architectures.


U-Net architectures are widely used for the segmentation of medical images with many papers proposing small changes to the basic U-Net architecture to improve the overall performance. `ModularUNet` allows to quickly try out multiple differnt building blocks with minimal code overhead. For example a `UResNet18` like model can be build with the following commands:  

At first create an abstract class, defining the basic building blocks. For a UResNet we want a `BasicResBlock` without bottlenecks in the encoder, a simpl double convolution after the encoder and simple `UNetBlock`s in the decoder. As final layer for creating the segmentation maps, we take a simple convolutional layer without normalization or activation layer. 

```
class UResNet(ModularUNet):    
    def encoder_layer(self, **kwargs): return BasicResBlock(**kwargs)
    def middle_layer(self, **kwargs): return DoubleConv(**kwargs)
    def skip_layer(self, **kwargs): return nn.Identity()
    def decoder_layer(self, **kwargs): return UnetBlock(**kwargs)
    def extra_after_decoder_layer(self, **kwargs): return nn.Identity()
    def final_layer(self, **kwargs): return ConvLayer(act=None, norm=None, **kwargs)
```

The abstract `UResNet` class can now be subclassed to specify a `UResNet18`.  

We want the number of channels after the first block (stem) to be 32, then gradually increase to 64, 128, 256, and 512.   
The `kernel_size` of the first Convlayer in the `BasicResBlock` should always be 3, except for the stem where it is 5.  
`stride` is kept constant and `padding` will be automatically be calculated for each `ConvLayer` in dependece to the `kernel_size` .        
To create a ResNet18-like architecture, we want to stack two `BasicResBlock` for each network block, creating `n_block` building blocks in the encoder


```
class UResNet18(UResNet):
    " UNet with ResNet18-like Backbone "
    channels = 32, 64, 128, 256, 512
    kernel_size = 5, 3, 3, 3, 3
    stride = 2, 2, 2, 2, 2
    padding = 'auto', 'auto', 'auto', 'auto', 'auto'
    n_layers = 1, 2, 2, 2, 2
    n_blocks = 5
```

```
uresnet18 = UResNet18(in_c = 3, n_classes = 2)
test_forward(uresnet18)
```

To change some components to the `UResNet18` one can either subclass or monkey patch additonal functonality. 

```
from fastcore.dispatch import patch
@patch
def decoder_layer(self:UResNet18, **kwargs): 
    return UnetBlock(spatial_attention=True, **kwargs)
```

```
uresnet18_with_attention = UResNet18(in_c = 3, n_classes = 2)
test_forward(uresnet18_with_attention)
```

```
assert not hasattr(uresnet18.decoder_block_1, 'sa')
assert hasattr(uresnet18_with_attention.decoder_block_1, 'sa')
```
