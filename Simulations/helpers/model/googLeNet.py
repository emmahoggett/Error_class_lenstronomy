import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
import ssl

import torch.utils.model_zoo as model_zoo
torch.manual_seed(0)

class GoogLeNet(nn.Module):
    """
    
        Neural network structure :

        (conv1):
            (0): Conv2d(in_channels, ch1x1, kernel_size=1)
            (1): BatchNorm2d(out_channels, eps=1e-03)
            (2): ReLU(inplace=True)
        
        (branch2):
            (0): Conv2d(in_channels, ch1x1, kernel_size=1)
            (1): BatchNorm2d(out_channels, eps=1e-03)
            (2): ReLU(inplace=True)
            (3): Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
            (4): BatchNorm2d(out_channels, eps=1e-03)
            (5): ReLU(inplace=True)
            
        (branch3):
            (0): Conv2d(in_channels, ch1x1, kernel_size=1)
            (1): BatchNorm2d(out_channels, eps=1e-03)
            (2): ReLU(inplace=True)
            (3): Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
            (4): BatchNorm2d(out_channels, eps=1e-03)
            (5): ReLU(inplace=True)
            
        (branch4):
            (0): MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
            (1): Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
            (2): BatchNorm2d(out_channels, eps=1e-03)
            (3): ReLU(inplace=True)
        
        (out):
            (0): Cat ([branch1, branch2, branch3, branch4], dim=1)
    
    """
    
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, global_params=None):
        """
        
        :param global_params  : tuple subclass, a tuple subclass that contain parameters for the entire model (stem, all blocks, and head) - default : None
        """
        super(GoogLeNet, self).__init__()

        if global_params.blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = global_params.aux_logits
        self.transform_input = global_params.transform_input

        self.conv1 = conv_block(global_params.num_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if global_params.aux_logits:
            self.aux1 = inception_aux_block(512, global_params.num_classes)
            self.aux2 = inception_aux_block(528, global_params.num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=global_params.dropout_rate)
        self.fc = nn.Linear(1024, global_params.num_classes)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, final tensor, and two auxiliary outputs
        """
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if aux_defined:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = F.sigmoid(self.fc(x))
        # N x 1000 (num_classes)
        return x, aux2, aux1

    def extract_features(self, inputs : torch.Tensor)-> torch.Tensor:
        """ Returns output of the final convolution layer """
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        return x



class Inception(nn.Module):
    """
    Inception model with dimension reduction.
    
    Neural network structure :

        (branch1):
            (0): Conv2d(in_channels, ch1x1, kernel_size=1)
            (1): BatchNorm2d(out_channels, eps=1e-03)
            (2): ReLU(inplace=True)
        
        (branch2):
            (0): Conv2d(in_channels, ch1x1, kernel_size=1)
            (1): BatchNorm2d(out_channels, eps=1e-03)
            (2): ReLU(inplace=True)
            (3): Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
            (4): BatchNorm2d(out_channels, eps=1e-03)
            (5): ReLU(inplace=True)
            
        (branch3):
            (0): Conv2d(in_channels, ch1x1, kernel_size=1)
            (1): BatchNorm2d(out_channels, eps=1e-03)
            (2): ReLU(inplace=True)
            (3): Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
            (4): BatchNorm2d(out_channels, eps=1e-03)
            (5): ReLU(inplace=True)
            
        (branch4):
            (0): MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
            (1): Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
            (2): BatchNorm2d(out_channels, eps=1e-03)
            (3): ReLU(inplace=True)
        
        (out):
            (0): Cat ([branch1, branch2, branch3, branch4], dim=1)
    
    
    """
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels:int, ch1x1:int, ch3x3red:int, ch3x3:int, ch5x5red:int, ch5x5:int, pool_proj:int, conv_block=None)->None:
        """
        
        :param in_channels : int, previous layer output size
        :param ch1x1       : int, number of output channels for the 1x1 convolution branch
        :param ch3x3red    : int, output channels' number of dimension reductions for the 3x3 convolution branch
        :param ch3x3       : int, output channels' number of 3x3 convolution for the 3x3 convolution branch
        :param ch5x5red    : int, output channels' number of dimension reductions for the 5x5 convolution branch
        :param ch5x5       : int, output channels' number of 5x5 convolution for the 5x5 convolution branch
        :param pool_proj   : int, output channels' number of dimension reductions for the 3x3 max pool branch
        :param conv_block  : nn.Module, 2D convolutionnal layer in pytorch framework
        """
        
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: torch.Tensor):
        """
        
        :param x : torch.Tensor, 2D tensor input of the inception layer
        :return  : sequence, sequence of the four branch output's tensors
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self,  x: torch.Tensor):
        """
        
        :param x : torch.Tensor, 2D tensor input of the inception layer
        :return  : torch.Tensor, concatenation of the four branch of the inception layer.
        """
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """
    Build the classifier neural network auxiliary outputs in an inception layer.
    
    Neural network structure :

        (adaptive_avg_pool2d):
            (0): AdaptiveAvgPool2d(output_size = 4)

        (conv):
            (0): Conv2d(in_channels, 128, kernel_size=1)

        (fc1):
            (0): Linear(in_features=2048, out_features=1024)
            (1): ReLU(inplace=True)
            (2): Drop

        (fc2):
            (0): Linear(in_features=1024, out_features=num_classes)
            (1): ReLU(inplace=True)
            (2): Sigmoid()
    """

    def __init__(self, in_channels:int, out_channels:int, conv_block=None):
        """
        
        :param in_channels  : int, number of input channels
        :param out_channels : int, number of classes
        :param conv_block   : 
        """
        
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, out_channels)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        
        :param x : torch.Tensor, 2D tensor input of the inception layer
        :return  : torch.Tensor
        """
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = F.sigmoid(self.fc2(x))
        # N x 3 (num_classes)

        return x


class BasicConv2d(nn.Module):
    """
    Build 2D convolutionnal layer with batch normalization and relu activation function.
    The generated output of this neural network is a 2D tensor.
    
    Neural network structure :

        (basic_conv2d):
            (0): Conv2d(1, out_channels)
            (2): BatchNorm2d(out_channels, eps=1e-03)
            (1): ReLU(inplace=True)
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) ->None:
        """
        
        :param in_channels  : int, number of input channels
        :param out_channels : int, number of output channels
        """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        
        :param x  : tensor, 2D tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    "num_classes", "aux_logits", "transform_input",
    "blocks", "dropout_rate", "image_size", "num_channel"
])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

def googlenet(aux_logits, transform_input, blocks,
              image_size, num_channel, num_classes, dropout_rate=0.2):
    """ 
    Creates a googlenet_pytorch model. 
    
    :param aux_logits      :
    :param transform_input :
    :param blocks          : 
    :param image_size      : 
    :param num_channel     : 
    :param dropout_rate    : float, dropout rate
    :return                : tuple subclass, a tuple subclass named GlobalParams with parameters for the entire model (stem, all blocks, and head)
    """

    global_params = GlobalParams(
        aux_logits=aux_logits,
        transform_input=transform_input,
        blocks=blocks,
        image_size=image_size,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        num_channel=num_channel,
    )

    return global_params

