import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class CustomAlexNet(nn.Module):
    def __init__(self, input_size: int, num_classes:int):
        """
        
        :param input_size  : int, number of channels in the image
        :param num_classes : int, number of labels to classify
        """
        super(CustomAlexNet, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_size, 96, kernel_size=10, stride=1),
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
    

class CustomVGG16(nn.Module):
    def __init__(self, input_size:int = 1, num_classes:int = 3):
        """
        
        :param input_size  : int, number of channels in the image
        :param num_classes : int, number of labels to classify
        """
        super(CustomVGG16, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_size, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding = 24)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )

        
    def forward(self, x: torch.Tensor):
        """
        
        :param x : image tensor
        :return  : forward neural network pass
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = x.view(x.size(0), 512*7*7)
        x = self.classifier(x)
        return x
 