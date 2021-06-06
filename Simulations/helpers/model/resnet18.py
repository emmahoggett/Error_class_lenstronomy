# this file contains the classes for residual neural networks
#          reference : pytorch - ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
torch.manual_seed(0)

class Conv2dAuto(nn.Conv2d):
    """
    2D convolutionnal neural network, with adapted padding.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

# 2D convolutionnal neural network - padding = 1, kernel = 3, bias = False
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)   

def activation_func(activation:str):
    """
    
    Dictionary with different activation functions.
    Defined activation function :
            - Rectified Linear Unit (ReLU)
            - Leaky Rectified Linear Unit (LeakyReLU)
            - Scaled Exponential Linear Unit (SELU)
            - None : A placeholder identity operator that is argument-insensitive.
            
    :param activation : str, name of the activation function
    :return           : nn.Module, activation layer
    """
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]        
    
    

class ResidualBlock(nn.Module):
    """
    Build a residual neural network block.
    
    
    """
    def __init__(self, in_channels:int, out_channels:int, activation:str='relu') -> None:
        """
        
        :param in_channels  : int, number of input channels in the image
        :param out_channels : int, number of labels to classify
        :param activation   : str, activation function used - default : activation = 'relu'
        """
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward residual block pass
        """
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self) -> bool:
        """
        
        :return : bool, check if the input channel and extended output channel are different.
        """
        return self.in_channels != self.out_channels
        
        
class ResNetResidualBlock(ResidualBlock):
    """
    Build a residual block.
    
    Structure of the residual block :
    
    (shortcut):
        (0): Conv2d(in_channels, expansion*out_channels, kernel_size=1, stride=downsampling, bias=False)
        (1): BatchNorm2d(expansion) if in_channels != expansion*out_channels
    """
    def __init__(self, in_channels:int, out_channels:int, expansion:int=1, downsampling:int=1, conv:partial=conv3x3, *args, **kwargs)->None:
        """
        
        :param in_channels  : int, number of input channels
        :param out_channels : int, number of output channels
        :param expansion    : int, expansion of the output channels
        :param downsampling : int, stride size of the 2D convolutionnal layer
        :param conv         : partial, 2D convolutionnal layer - default : 2D convolutionnal layer, kernel = 3, padding = 1, bias = False
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self)->int:
        """
        
        :return : int, expansion*out_channels
        """
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self)->bool:
        """
        
        :return : bool, check if the input channel and output channel are different
        """
        return self.in_channels != self.expanded_channels
        


def conv_bn(in_channels:int, out_channels:int, conv, *args, **kwargs):
    """
    Convolutionnal batch normalized layer.
    
    (conv_bn):
        (0): conv(in_channels, out_channels, *args, **kwargs)
        (1): nn.BatchNorm2d(out_channels)
    
    :param in_channels  : int, number of input channels
    :param out_channels : int, number of output channels
    :param conv         : nn.Conv2d, 2D convolutionnal layer
    :return             : nn.Sequential, 2D convolutionnal layer & batch normalization
    """
    # Convolutionnal layer combined with a batch normalisation
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block with an expansion = 1.
    
    Structure of ResNet block :
    
    (blocks):
        (0): conv_bn(in_channels, out_channels, conv=conv, bias=False, stride=downsampling)
        (1): activation_func(activation)
        (2): conv_bn(out_channels, expanded_channels, conv=conv, bias=False)
    """
    expansion = 1
    def __init__(self, in_channels:int, out_channels:int, *args, **kwargs):
        """

        :param in_channels  : int, number of input channels
        :param out_channels : int, number of output channels
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    """
    ResNet BottleNeck Block with an expansion = 4.
    
    Structure ResNet BottleNeck Block:
    
    (blocks): 
        (0): conv_bn(in_channels, out_channels, conv, kernel_size=1)
        (1): activation_func(activation)
        (2): conv_bn(out_channels, out_channels, conv, kernel_size=3, stride=downsampling)
        (3): activation_func(activation)
        (4): conv_bn(out_channels, expanded_channels, conv, kernel_size=1),
    """
    expansion = 4
    def __init__(self, in_channels:int, out_channels:int, *args, **kwargs):
        """

        :param in_channels  : int, number of input channels
        :param out_channels : int, number of output channels
        """
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )    


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other.
    
    Structure ResNet Layer:
    
    (blocks):
        (0)        : block(in_channels , out_channels, downsampling=downsampling)
        (1 - (n-1)): block(out_channels * block.expansion, out_channels, downsampling=1)
    """
    def __init__(self, in_channels:int, out_channels:int, block=ResNetBasicBlock, n:int=1, *args, **kwargs):
        """

        :param in_channels  : int, number of input channels
        :param out_channels : int, number of output channels
        :param block        : nn.Sequential, residual block - default: block = ResNetBasicBlock
        :param n            : int, number of blocks in the ResNet Layer - default: n = 1
        """
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward residual layer pass
        """
        x = self.blocks(x)
        return x
        

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    
    Structure ResNet Encoder :
    (gate):
        (0): Conv2d(in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
        (1): BatchNorm2d(blocks_sizes[0])
        (2): activation_func(activation)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    (blocks):
        (0 - len(deepths)-1): ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, block=block)
                              *[ResNetLayer(in_channels * block.expansion,  out_channels, n=n, activation=activation, block=block) 
    
    """
    def __init__(self, in_channels:int=1, blocks_sizes:list=[64, 128, 256, 512], deepths:list=[2,2,2,2], 
                 activation:str='relu', block=ResNetBasicBlock, *args, **kwargs):
        """
        
        :param in_channels  : int, images' number of input channels - default : in_channels = 1
        :param blocks_sizes : list, channel sizes - default: blocks_sizes = [64, 128, 256, 512]
        :param deepths      : list, number of ResNetLayer in each iteration of a block - default: deepths = [2,2,2,2]
        :param activation   : str, activation function - default : activation = 'relu'
        :param block        : ResNetResidualBlock, residual block - default : block = ResNetBasicBlock
        """
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward residual network encoder pass
        """
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    
        
class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer. The output has a sigmoid activation function to 
    use a binary cross entropy criterion for multilabeling.
    
    Structure of the ResNet Decoder:
    
    (avg):
        (0): AdaptiveAvgPool2d((1, 1))
        (1): Linear(in_features, out_channels)
    
    """
    def __init__(self, in_features:int, out_channels:int) -> None:
        """
        
        :param in_features  : int, number of input channels
        :param out_channels : int, number of labels to classify
        """
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, out_channels)

    def forward(self, x:torch.Tensor):
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward residual network decoder layer pass
        """
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = F.sigmoid(x)
        return x


class ResNet(nn.Module):
    """
    Combine the encoder and the decoder.
    
    Structure of the ResNet module :
    
    (encoder):
        (0): ResNetEncoder(in_channels)
    
    (decoder):
        (0): ResnetDecoder(encoder.blocks[-1].blocks[-1].expanded_channels, out_channels)
    """
    def __init__(self, in_channels:int, out_channels:int, *args, **kwargs) -> None:
        """
        
        :param in_channels  : int, image channel size
        :param out_channels : int, number of output classes
        """
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, out_channels)
         
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward residual network pass
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet18(in_channels:int, out_channels:int, block=ResNetBasicBlock, *args, **kwargs) -> ResNet:
    """
    
    :param in_channels  : int, image channel size
    :param out_channels : int, number of output classes
    :param block        : ResNetResidualBlock, residual block - default : block = ResNetBasicBlock
    :return             : ResNet, residual network with 18 layers
    """
    return ResNet(in_channels, out_channels, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)
    
def resnet34(in_channels:int, out_channels:int, block=ResNetBasicBlock, *args, **kwargs)->ResNet:
    """
    
    :param in_channels  : int, image channel size
    :param out_channels : int, number of output classes
    :param block        : ResNetResidualBlock, residual block - default : block = ResNetBasicBlock
    :return             : ResNet, residual network with 34 layers
    """
    return ResNet(in_channels, out_channels, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet50(in_channels, out_channels, block=ResNetBottleNeckBlock, *args, **kwargs)->ResNet:
    """
    
    :param in_channels  : int, image channel size
    :param out_channels : int, number of output classes
    :param block        : ResNetResidualBlock, residual block - default : block = ResNetBasicBlock
    :return             : ResNet, residual network with 50 layers
    """
    return ResNet(in_channels, out_channels, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet101(in_channels, out_channels, block=ResNetBottleNeckBlock, *args, **kwargs)->ResNet:
    """
    
    :param in_channels  : int, image channel size
    :param out_channels : int, number of output classes
    :param block        : ResNetResidualBlock, residual block - default : block = ResNetBasicBlock
    :return             : ResNet, residual network with 101 layers
    """
    return ResNet(in_channels, out_channels, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)

def resnet152(in_channels, out_channels, block=ResNetBottleNeckBlock, *args, **kwargs)->ResNet:
    """
    
    :param in_channels  : int, image channel size
    :param out_channels : int, number of output classes
    :param block        : ResNetResidualBlock, residual block - default : block = ResNetBasicBlock
    :return             : ResNet, residual network with 152 layers
    """
    return ResNet(in_channels, out_channels, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)
