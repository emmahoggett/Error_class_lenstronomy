# this file contains the classes for basic neural networks

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)


class CNNNetBasic(nn.Module):
    """
    Basic neural network class for residual maps. The output returns a sigmoid of size out_channels.
    
    Neural network structure :
    
    (conv_base):
        (0): Conv2d(in_channels, 6, kernel_size=(5, 5), stride=(1, 1))
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        
    
    (classifier): 
        (0): Linear(in_features=2704, out_features=120, bias=True)
        (1): Linear(in_features=120, out_features=84, bias=True)
        (2): Linear(in_features=84, out_features=out_channels, bias=True)
        (3): Sigmoid()
    """
    def __init__(self, in_channels: int = 1, out_channels:int = 2) -> None:
        """
        
        :param in_channels  : int, image channel size - default : in_channels = 1
        :param out_channels : int, number of output classes - default : out_channels = 2
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, bias = False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias = False)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.typenet = 'conv'
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : torch.Tensor, image tensor
        :return  : torch.Tensor, forward neural network pass
        """
        x = self.pool(F.selu(self.conv1(x)))
        x = self.pool(F.selu(self.conv2(x)))
        x = x.view(x.size(0), 16 * 13 * 13)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
        
class TabularNetBasic(nn.Module):
    """
    Basic neural network class for metadata. The output returns a sigmoid of size out_channels.
    
    Neural network structure :
    
    (classifier): 
        (0): Linear(in_features=meta_channels, out_features=16, bias=True)
        (1): Linear(in_features=16, out_features=8, bias=True)
        (2): Linear(in_features=8, out_features=3, bias=True)
        (3): Sigmoid()
    """
    def __init__(self, meta_channels: int, out_channels:int = 2) -> None:
        """
        
        :param meta_channels : int, metadata parameters size
        :param out_channels  : int, number of output classes - default : out_channels = 2
        """
        super(TabularNetBasic, self).__init__()

        self.fc1 = nn.Linear(meta_channels, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, out_channels)
        self.typenet = 'meta'
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : torch.Tensor, metadata tensor
        :return  : torch.Tensor, forward neural network pass
        """

        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x

class TabularCNNNetBasic(nn.Module):
    """
    Basic neural network class for residual maps and metadata. The output returns a sigmoid of size 3.
    
    Neural network structure :
    
    (metadata): 
        (0): Linear(in_features=meta_channels, out_features=16, bias=True)
        (1): Linear(in_features=16, out_features=8, bias=True)
    
    (conv_base): 
        (0): Conv2d(in_channels, 6, kernel_size=(5, 5), stride=(1, 1))
        (1): MaxPool2d(kernel_size=2, stride=2)
        (2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        (3): Linear(in_features=2704, out_features=120, bias=True)
        (4): Linear(in_features=120, out_features=84, bias=True)
    
    (classifier): 
        (0): Linear(in_features=92, out_features=60, bias=True)
        (1): Linear(in_features=60, out_features=out_channels, bias=True)
        (2): Sigmoid()
    """
    def __init__(self, meta_channels: int, in_channels:int = 1, out_channels:int = 2) -> None:
        """
        
        :param meta_channels : int, metadata parameters size 
        :param in_channels   : int, image channel size - default : in_channels = 1
        :param out_channels  : int, number of output classes - default : out_channels = 2
        """
        
        super(TabularCNNNetBasic, self).__init__()

        self.fc1_data = nn.Linear(meta_channels, 16)
        self.fc2_data = nn.Linear(16, 8)

        self.conv1_img = nn.Conv2d(in_channels, 6, kernel_size=5, bias = False)
        self.pool_img = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_img = nn.Conv2d(6, 16, kernel_size=5, bias = False)
        self.fc1_img = nn.Linear(16 * 13 * 13, 120)
        self.fc2_img = nn.Linear(120, 84)

        self.fc1 = nn.Linear(8 + 84, 60)
        self.fc2 = nn.Linear(60, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.typenet = 'convXmeta'
        

    def forward(self, img: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        
        :param img  : torch.Tensor, image tensor
        :param data : torch.Tensor, metadata tensor
        :return     : torch.Tensor, forward neural network pass
        """

        data = F.selu(self.fc1_data(data))
        data = F.selu(self.fc2_data(data))

        img = self.pool_img(F.selu(self.conv1_img(img)))
        img = self.pool_img(F.selu(self.conv2_img(img)))
        img = img.view(img.size(0), 16 * 13 * 13)
        img = F.selu(self.fc1_img(img))
        img = F.selu(self.fc2_img(img))

        x = torch.cat((img, data), dim=1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x
        



