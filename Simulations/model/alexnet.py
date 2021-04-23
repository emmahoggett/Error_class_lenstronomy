# this file contains the class for the Alexnet neural network

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetResidual(nn.Module):
    """
    Alex neural network structure with a multilabelling output. The output returns a sigmoid of size 3.
    
    Neural network structure :
    
    (conv_base):
        (0): Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4))
        (1): ReLU(inplace=True)
        (2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (4): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (5): ReLU(inplace=True)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (8): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
    
    (classifier): 
        (0): Linear(in_features=6400, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=4096, out_features=4096, bias=True)
        (3): ReLU(inplace=True)
        (4): Linear(in_features=4096, out_features=3, bias=True)
        (5): Sigmoid()
    """
    def __init__(self, input_size: int, num_classes:int):
        """
        
        :param input_size  : int, number of channels in the image
        :param num_classes : int, number of labels to classify
        """
        super(AlexNetResidual, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_size, 96, kernel_size=11, stride=4),
            nn.ReLU(), nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(), nn.BatchNorm2d(256), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x: torch.Tensor):
        """
        
        :param x : image tensor
        :return  : forward neural network pass
        """
        x = self.conv_base(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        