# this file contains the classes for basic neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNNetBasic(nn.Module):
    """
    Basic neural network class for residual maps. The output returns a sigmoid of size 3.
    
    Neural network structure :
    
    (conv_base):
        (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        
    
    (classifier): 
        (0): Linear(in_features=2704, out_features=120, bias=True)
        (1): Linear(in_features=120, out_features=84, bias=True)
        (2): Linear(in_features=84, out_features=3, bias=True)
        (3): Sigmoid()
    """
    def __init__(self, input_size: int = 1, num_classes:int = 3)-> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.typenet = 'conv'
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : image tensor
        :return  : forward neural network pass
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
    Basic neural network class for metadata. The output returns a sigmoid of size 3.
    
    Neural network structure :
    
    (classifier): 
        (0): Linear(in_features=11, out_features=16, bias=True)
        (1): Linear(in_features=16, out_features=8, bias=True)
        (2): Linear(in_features=8, out_features=3, bias=True)
        (3): Sigmoid()
    """
    def __init__(self, meta_size: int = 11, num_classes:int = 2) -> None:
        super(TabularNetBasic, self).__init__()

        self.fc1 = nn.Linear(meta_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, num_classes)
        self.typenet = 'meta'
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : metadata tensor
        :return  : forward neural network pass
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
        (0): Linear(in_features=11, out_features=16, bias=True)
        (1): Linear(in_features=16, out_features=8, bias=True)
    
    (conv_base): 
        (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        (1): MaxPool2d(kernel_size=2, stride=2)
        (2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        (3): Linear(in_features=2704, out_features=120, bias=True)
        (4): Linear(in_features=120, out_features=84, bias=True)
    
    (classifier): 
        (0): Linear(in_features=92, out_features=60, bias=True)
        (1): Linear(in_features=60, out_features=3, bias=True)
        (2): Sigmoid()
    """
    def __init__(self, meta_size: int = 11, img_size:int = 1, num_classes:int = 2) -> None:
        super(TabularCNNNetBasic, self).__init__()

        self.fc1_data = nn.Linear(meta_size, 16)
        self.fc2_data = nn.Linear(16, 8)

        self.conv1_img = nn.Conv2d(img_size, 6, kernel_size=5)
        self.pool_img = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_img = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1_img = nn.Linear(16 * 13 * 13, 120)
        self.fc2_img = nn.Linear(120, 84)

        self.fc1 = nn.Linear(8 + 84, 60)
        self.fc2 = nn.Linear(60, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.typenet = 'convXmeta'
        

    def forward(self, img: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        
        :param img  : image tensor
        :param data : metadata tensor
        :return     : forward neural network pass
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
        



