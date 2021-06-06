# this file contains the class for the VGG neural network
#          reference : pytorch - VGG

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

from typing import Union, List, Dict, Any, cast

class VGG(nn.Module):
    """
    VGG network structure with a multilabelling output. The output returns a sigmoid of size out_channels.
    
    Neural network structure :
    
    (features):
        Layers combinations that depends on which VGG structure selected
    
    (avgpool):
        (0): AdaptiveAvgPool2d(kernel=7)
    
    (classifier): 
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.65, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.65, inplace=False)
        (6): Linear(in_features=4096, out_features=out_channels, bias=True)
        
    (sigmoid):
        (0): Sigmoid()

    """
    def __init__(self, features: nn.Module, out_channels: int = 2) -> None:
        """
        
        :param features     : nn.Module, convolutionnal layers structures
        :param out_channels : int, number of labels to classify - default : out_channels = 2
        """
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_channels),
        )
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward neural network pass
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.sigmoid(self.classifier(x))
        return x
    

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels:int = 1) -> nn.Sequential:
    """
    
    :param cfg         : List[Union[str, int]], list of string and integers that define the structure of the convolutionnal neural network
    :param batch_norm  : bool, batch normalisation - default : batch_norm = False
    :param in_channels : int, number of channels in the image - default : in_channels = 1
    :return            : nn.Sequential, layers for the 2D convolutionnal neural network for VGG model
    """
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias = False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# Dictionnary of VGG11 & VGG16
cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
    
    
def _vgg(cfg: str, batch_norm: bool, in_channels:int, **kwargs: Any) -> VGG:
    """
    
    :param cfg         : str, select the structure of convolutionnal neural network in cfgs dictionnary
    :param batch_norm  : bool, batch normalisation activation
    :param in_channels : int, number of channels in the image
    :return            : VGG, VGG model
    """
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels = in_channels), **kwargs)
    return model


def vgg11(in_channels:int, **kwargs: Any) -> VGG:
    """
    
    VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    
    
    :param in_channels : int, number of channels in the image
    :return            : VGG, VGG model
    """
    return _vgg('A', False, in_channels, **kwargs)


def vgg11_bn(in_channels:int, **kwargs: Any) -> VGG:
    """
    
    VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    
    :param in_channels : int, number of channels in the image
    :return            : VGG, VGG model
    """
    return _vgg('A', True, in_channels, **kwargs)


def vgg16(in_channels:int, **kwargs: Any) -> VGG:
    """
    
    VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    
    :param in_channels : int, number of channels in the image
    :return            : VGG, VGG model
    """
    return _vgg( 'B', False, in_channels, **kwargs)


def vgg16_bn(in_channels:int, **kwargs: Any) -> VGG:
    """
    
    VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    
    :param in_channels : int, number of channels in the image
    :return            : VGG, VGG model
    """
    return _vgg('B', True, in_channels, **kwargs)


 