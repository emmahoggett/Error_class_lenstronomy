import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetResidual(nn.Module):
    def __init__(self, input_size: int = 1, num_classes:int = 3):
        super(AlexNetResidual, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_size, 96, kernel_size=11, stride=4),
            nn.ReLU(True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
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
        self.typenet = 'conv'
        
        
    def forward(self, x: torch.Tensor):
        x = self.conv_base(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        