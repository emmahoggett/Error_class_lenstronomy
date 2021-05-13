# this file contains the class for the Alexnet neural network

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    Alex neural network structure with a multilabelling output. The output returns a sigmoid of size 3.
    If a parameter is not mentionned, it implies that the default parameter defined by Pytorch is taken.
    
    Neural network structure :
    
    (conv_base):
        (0): Conv2d(input_size, 64, kernel_size=11, stride=4)
        (1): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=3, stride=2)
        (4): Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        (5): ReLU(inplace=True)
        (7): MaxPool2d(kernel_size=3, stride=2)
        (8): Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        (9): ReLU(inplace=True)
        (10): Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        (13): ReLU(inplace=True)
        (14): MaxPool2d(kernel_size=3, stride=2)
        
    
    (classifier): 
        (0): Linear(in_features=6400, out_features=4096)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=4096, out_features=4096)
        (3): ReLU(inplace=True)
        (4): Linear(in_features=4096, out_features=num_classes)
        (5): Sigmoid()
    """
    def __init__(self, input_size: int, num_classes:int) -> None:
        """
        
        :param input_size  : int, number of channels in the image
        :param num_classes : int, number of labels to classify
        """
        super(AlexNet, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        :param x : image tensor
        :return  : forward neural network pass
        """
        x = self.conv_base(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
        