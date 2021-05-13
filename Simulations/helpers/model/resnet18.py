import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class Conv2dAuto(nn.Conv2d):
    """
    Compute padding based on the kernel size.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)   



def activation_func(activation):
    """
    Dictionary with different activation functions.
    Defined activation function :
            - Rectified Linear Unit (ReLU)
            - Leaky Rectified Linear Unit (LeakyReLU)
            - Scaled Exponential Linear Unit (SELU)
            - None : A placeholder identity operator that is argument-insensitive.
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
    Inputs :
            - in_channels   :  Number of input channels.
            - blocks_sizes  :  Number of output channels.
            - activation    :  Activation function, the default activation function is ReLU.
                               Use the dictionary to change the activation function ('leaky_relu',
                               'selu', 'none')
    """
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        #Check if the input channel and output channel are different.
        return self.in_channels != self.out_channels
        
        
class ResNetResidualBlock(ResidualBlock):
    """
    Build a residual block 
    Inputs :
            - in_channels   : Number of input channels.
            - out_channels  : Number of output channels.
            - expansion     :
            - downsampling  :
            - conv          : 
    """
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        #
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        #Check if the input channel and output channel are different.
        return self.in_channels != self.expanded_channels
        


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    # Convolutionnal layer combined with a batch normalisation
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    Inputs :
            - in_channels : Number of input channels.
            - out_channels: Number of output channels.
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    """
    
    Inputs :
            - in_channels : Number of input channels.
            - out_channels: Number of output channels.
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
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
    A ResNet layer composed by `n` blocks stacked one after the other
    Inputs :
            - in_channels : Number of input channels.
            - out_channels: Number of output channels.
            - block       :
            - n           : Number of stacked block. The default value is n=1.
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
        

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    Inputs :
            - in_channels   :  Number of input channels. The channel is set by default to 1.
            - blocks_sizes  :  
            - deepths       :  
            - activation    :  Activation function, the default activation function is ReLU.
                               Use the dictionary to change the activation function ('leaky_relu',
                               'selu', 'none')
    """
    def __init__(self, in_channels=1, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
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
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    
        
class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer. The output has a sigmoid activation function to 
    use a binary cross entropy criterion for multilabeling.
    Inputs :
        - in_channels   :  Number of input channels. This value correspond to the output length of
                           the flatten convolutionnal output.
        - n_classes     :  Number of output labels.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = F.sigmoid(x)
        return x


class ResNet(nn.Module):
    """
    Combine the encoder and the decoder.
    Inputs :
            - in_channels   :  Number of input channels. The channel is set by default to 1.
            - n_classes     :  Number of output labels.
    """
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        self.typenet = 'conv'
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet18maps(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)
    
def resnet34maps(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet50maps(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet101maps(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)

def resnet152maps(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)