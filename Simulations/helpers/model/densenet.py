import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

from collections import OrderedDict
from torch import Tensor
from typing import Any

class _Transition(nn.Sequential):
    """
    Transition layers after a dense block.
    
    Transition layers structure :
    
        (norm): 
            BatchNorm2d(num_input_features)
        (relu): 
            ReLU(inplace=True)
        (conv): 
            Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1)
        (pool): 
            AvgPool2d(kernel_size=2, stride=2)
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        
        :param num_input_features  : int, input size features
        :param num_output_features : int, output size features
        """
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseLayer(nn.Module):
    """
    Define one dense layer.
    
    Dense layer stucture :
        (norm1):
            BatchNorm2d(num_input_features)
        (relu1):
            ReLU(inplace=True)
        (conv1):
            Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)
        (norm2):
             BatchNorm2d(bn_size*growth_rate)
        (relu2):
            ReLU(inplace=True)
        (conv2):
            Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
            
    """
    def __init__(self, num_input_features:int, growth_rate:int, bn_size:int, drop_rate:float, memory_efficient:bool=False):
        """
        
        :param num_input_features  : int, number of input channels
        :param growth_rate         : int, number of output channels of dense layers
        :param bn_size             : int, intermediate layer size factor such that interm_size = bn_size*growth_rate
        :param drop_rate           : float, dropout rate
        :param memory_efficient    : bool, if True, the model use checkpointing which improve memory management, yet increase training time - default : False
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: torch.Tensor):
        """
        
        :param inputs : tensor, input list of tensor 
        :return       : tensor, concatenate input tensors 
        """
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input): 
        """
        
        :param inputs : tensor, input list of tensor 
        :return       : tensor, concatenate input tensors 
        """
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2
    """
    Dense block module dictionnary.
    
    Dense net structure repeated num_layers times:
        (DenseLayer):
             _DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, memory_efficient=memory_efficient)
            
    """
    def __init__(self, num_layers: int, num_input_features: int, bn_size:int, growth_rate:int, drop_rate:float, memory_efficient:bool=False):
        """
        
        :param num_layers         : int, number of dense layer in the dense block
        :param num_input_features : int, number of input channels
        :param bn_size            : int, intermediate layer size factor such that interm_size = bn_size*growth_rate
        :param growth_rate        : int, number of output channels of dense layers 
        :param drop_rate          : float, dropout rate between layers
        :param memory_efficient   : bool, if True, the model use checkpointing which improve memory management, yet increase training time - default : False
        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        """
        
        :param init_features : tensor, dense block input
        :return              : tensor, dense block output
        """
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
        



class DenseNet(nn.Module):
    """
    Dense neural network module.
    
    Dense neural network struct
    
    """
    def __init__(self, growth_rate: int=32, block_config: tuple=(6, 12, 24, 16),
                 num_init_features:int=64, bn_size:int=4, drop_rate:float=0, nb_chn:int = 1, num_classes:int=2, memory_efficient:bool=False):
        """
        
        :param growth_rate       : int, number of output channels of dense layers - default : 32
        :param block_config      : tuple, number of dense layer in the four dense block - default : (6, 12, 24, 16)
        :param num_init_features : int, number of input channels - default : 64
        :param bn_size           : int, intermediate layer size factor such that interm_size = bn_size*growth_rate - default : 4
        :param drop_rate         : float, dropout rate between layers - default : 0
        :param nb_chn            : int, number of channels in the image - default : 1
        :param num_classes       : int, number of labels for classification - default: 2
        :param memory_efficient  : bool, if True, the model use checkpointing which improve memory management, yet increase training time - default : False
        """
        super(DenseNet, self).__init__()

        # Convolution and pooling part from table-1
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(nb_chn, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple denseblocks based on config 
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to 
                # downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        
        :param x : tensor, image
        :return  : tensor, forward neural network pass
        """
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = F.sigmoid(self.classifier(out))
        return out
        
def _densenet(arch:str, growth_rate:int, block_config:tuple, num_init_features:int, pretrained:bool, progress:bool,
              input_size:int, num_classes:int , **kwargs):
    """
    Return the DenseNet module with a given configuration. 
    
    :param arch              : string, name of the achitecture
    :param growth_rate       : int, number of output channels of dense layers
    :param block_config      : tuple, number of dense layer in the four dense block
    :param num_init_features : int, number of input channels
    :param pretrained        : bool, if true, returns a model pre-trained on ImageNet
    :param progress          : bool, if true, displays a progress bar of the download to stderr
    :param input_size        : int, number of input channels
    :param num_classes       : int, number of labels for classification
    :return                  : module, dense net module 
    """
    model = DenseNet(growth_rate, block_config, num_init_features, nb_chn = input_size,  num_classes = num_classes, **kwargs)
    return model

def densenet121(pretrained:bool=False, progress:bool=True, input_sz:int = 1, num_classes:int = 2,**kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, input_sz, num_classes,
                     **kwargs)        

def densenet161(pretrained: bool = False, progress: bool = True, input_sz:int = 1, num_classes:int = 2, **kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress, input_sz, num_classes,
                     **kwargs)