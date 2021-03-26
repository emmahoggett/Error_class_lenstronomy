import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16Residual(nn.Module):
    def __init__(self, input_size = 1, num_classes=3):
        super(VGG16Residual, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_size, 64,kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
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
        self.typenet = 'conv'
        
    def netype(self):
        return (self.typenet)
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(x.size(0), 512*7*7)
        x = self.classifier(x)
        return x
        
class VGG16ResidualMetadata(nn.Module):
    def __init__(self, input_size=1, metadata_size = 7, num_classes=3):
        super(VGG16ResidualMetadata, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_size, 64,kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.fc_data = nn.Sequential(
            nn.Linear(metadata_size, 64),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),   

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),         
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7 + 512, 4096),
            nn.ReLU(True), nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True), nn.Dropout(p=0.65),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )
        self.typenet = 'convXmeta'
        
    def netype(self):
        return (self.typenet)
        
    def forward(self, img, meta):
        img = self.block_1(img)
        img = self.block_2(img)
        img = self.block_3(img)
        img = self.block_4(img)
        img = self.block_5(img)
        img = img.view(img.size(0), 512*7*7)

        meta = self.fc_data(meta)

        x = torch.cat((img, data), dim=1)
        x = self.classifier(x)
        return x