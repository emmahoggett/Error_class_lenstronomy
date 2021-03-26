import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetResidual(nn.Module):
    def __init__(self, input_size = 1, num_classes=3):
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
        
    def netype(self):
        return (self.typenet)
        
    def forward(self, x):
        x = self.conv_base(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
class AlexNetResidualMetadata(nn.Module):
    def __init__(self, input_size = 1, metadata_size = 7, num_classes=3):
        super(AlexNetResidualMetadata, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_size, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc_data = nn.Sequential(
            nn.ReLU(metadata_size, 64),nn.ReLU(True),

            nn.Linear(64, 192), nn.ReLU(True),
            nn.Linear(192, 384), nn.ReLU(True),
            nn.Linear(384, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True),             
        )
        self.fc_base = nn.Sequential(
            nn.Linear(256*5*5+256, 4096),
            nn.ReLU(True),
            
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )
        self.typenet = 'convXmeta'
        
    def netype(self):
        return (self.typenet)
        
    def forward(self, img, data):
        img = self.conv_base(img)
        img = torch.flatten(img, 1)

        data = self.fc_data(data)
        
        x = torch.cat((img, data), dim=1)
        x = self.fc_base(x)

        return x