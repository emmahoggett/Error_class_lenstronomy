# this file contains the class for the DenseNet neural network
#          reference : pytorch - SqueezeNet

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any
torch.manual_seed(0)


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1', 'squeezenet_custom']



class Fire(nn.Module):
    """
    Class for the Fire layer in SqueezeNet neural network.
    
    Fire layer structure. The output is the concatenation of the expand1x1 and the expand3x3 layer:
    
    (squeeze):
        (0): Conv2d(inplanes, squeeze_planes, kernel_size=1)
        (1): ReLU(inplace=True)
        
    (expand1x1):
        (0): Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        (1): ReLU(inplace=True)
    
    (expand3x3):
        (0): Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3)
        (1): ReLU(inplace=True)
    """
    def __init__( self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        """
        
        :param inplanes         : int, input number of channels
        :param squeeze_planes   : int, squeeze layer output size
        :param expand1x1_planes : int, output size of the 1x1 kernel convolutionnal layer
        :param expand3x3_planes : int, output size of the 3x3 kernel convolutionnal layer
        """
        
        
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : torch.Tensor, 2D tensor
        :return  : torch.Tensor, forward neural network pass
        """
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    """
    Class that build the SqueezeNet neural network with a multilabelling output. The output returns a sigmoid of size 3.
    The structure between the versions is similar, the only parameter that changes is the 2D convolutional section named
    "features".
    
    Neural network structure for the 1.1 SqueezeNet:
    
    (features):
        (0): Conv2d(in_channels, 64, kernel_size=3, stride=2)
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        (3): Fire(64, 16, 64, 64)
        (4): Fire(128, 16, 64, 64)
        (5): MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        (6): Fire(128, 32, 128, 128)
        (7): Fire(256, 32, 128, 128)
        (8): MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        (9): Fire(256, 48, 192, 192)
        (10): Fire(384, 48, 192, 192)
        (11): Fire(384, 64, 256, 256)
        (12): Fire(512, 64, 256, 256)
        
    (classifier):
        (0): Dropout(p=0.5)
        (1): Conv2d(512, self.num_classes, kernel_size=1)
        (2): ReLU(inplace=True)
        (3): AdaptiveAvgPool2d((1, 1))
        (4): Sigmoid()
    
    """

    def __init__( self, version: str = '1_0', in_channels: int = 1, num_classes: int = 2) -> None:
        """
        
        :param version     : str, version of the SqueezeNet - default : version = '1_0'
        :param in_channels : int, number of input channels in the image - default : in_channels = 1
        :param num_classes : int, number of output labels - default : num_classes = 2
        """
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        elif version == 'custom':
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=6, stride=1, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
            
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0, 1_1 or custom expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward neural network pass
        """
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, in_channels: int, out_channels: int , **kwargs: Any) -> SqueezeNet:
    """
    
    :param version      : str, version of the SqueezeNet
    :param in_channels  : int, number of input channels in the image
    :param out_channels : int, number of output labels
    :return             : SqueezeNet, squeeze net model
    """
    return SqueezeNet(version, in_channels=in_channels, num_classes= num_classes,  **kwargs)
    


def squeezenet1_0(in_channels: int=1, out_channels: int=2, **kwargs: Any) -> SqueezeNet:
    """
    
    SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    
    :param in_channels  : int, number of input channels in the image - default : in_channels = 1
    :param out_channels : int, number of output labels - default : out_channels = 2
    :return             : SqueezeNet, SqueezeNet model for the version 1.0 
    """
    return _squeezenet('1_0', in_channels, out_channels, **kwargs)


def squeezenet1_1(in_channels: int=1, out_channels: int=2, **kwargs: Any) -> SqueezeNet:
    """
    
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    
    :param in_channels  : int, number of input channels in the image - default : in_channels = 1
    :param out_channels : int, number of output labels - default : out_channels = 2
    :return             : SqueezeNet, SqueezeNet model for the version 1.1
    """
    return _squeezenet('1_1', in_channels, out_channels, **kwargs)

def squeezenetcustom(in_channels: int=1, out_channels: int=2, **kwargs: Any) -> SqueezeNet:
    """
    
    :param in_channels  : int, number of input channels in the image - default : in_channels = 1
    :param out_channels : int, number of output labels - default : out_channels = 2
    :return             : SqueezeNet, SqueezeNet model for input image of (1,64,64)
    """
    return _squeezenet('custom', in_channels, out_channels, **kwargs)

