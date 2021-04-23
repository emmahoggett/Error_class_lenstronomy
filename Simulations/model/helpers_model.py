import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from model.baseline import CNNNetBasic, TabularNetBasic, TabularCNNNetBasic
from model.densenet import densenet161
from model.alexnet import AlexNetResidual
from model.resnet18 import resnet18maps
from model.vgg16 import VGG16Residual
from model.googLeNet import googlenet, GoogLeNet
from model.squeezeNet import squeezenet1_1

from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score


class NeuralNet:
    def __init__(self, net:str, optimizer:str, nb_chn:int = 1, nb_classes:int = 2, meta_in:int = 11):
        self.net_name = net
        self.net = self.CNN_dict(net, nb_chn, nb_classes, meta_in)
        self.final_net = self.net
        self.max_met = 0
        self.criterion = nn.BCELoss()
        self.optimizer = self.optimizers_dict(optimizer)
    
    def train (self, loader_train, resize_tp:str = 'Padding'):
        self.resize_tp = resize_tp
        for i, data in enumerate(loader_train, 0):
            inputs, meta, labels = data
            if self.net_name!='BasicCNN' and self.net_name!='BasicTabular' and  self.net_name!='BasicCNNTabular':
                inputs = self.resize_dict(inputs, self.resize_tp)
                
            # zero the parameter gradients
            self.optimizer.zero_grad()
            
             # forward + backward + optimize
            loss = 0
            if self.net_name=='BasicTabular':
                outputs = self.net(meta)
            elif self.net_name=='BasicCNNTabular':
                outputs = self.net(inputs,meta)
            else:
                outputs = self.net(inputs)
                
            if self.net_name == 'GoogleNet':
                for output in zip(outputs):
                    loss = loss + self.criterion(output, labels)      
            else:
                loss = self.criterion(outputs,labels)   
            loss.backward()
            self.optimizer.step()
            
            
    def test (self, loader_test, epoch, metric:str = 'accuracy'):
        with torch.no_grad():
            predictions = []
            targets = []
            for data in loader_test:
                images, meta, labels = data
                
                if self.net_name!='BasicCNN' and self.net_name!='BasicTabular' and  self.net_name!='BasicCNNTabular':
                    images = self.resize_dict(images, self.resize_tp)
                    
                    
                if self.net_name=='BasicTabular':
                    outputs = self.net(meta)
                elif self.net_name=='BasicCNNTabular':
                    outputs = self.net(images,meta)
                elif self.net_name=="GoogleNet":
                    outputs, aux_1, aux_2 = self.net(images)
                    outputs = (outputs + aux_1 + aux_2)/3
                else:
                    outputs = self.net(images)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
            result = self.calculate_metrics(np.round(np.array(predictions)), np.array(targets))
        self._update_(result[metric], epoch)
            
        return result[metric]
    
    def _update_(self, result, epoch):
        if result > self.max_met:
            self.max_met = result
            self.final_net = self.net
            self.opti_epoch = epoch
            print("epoch: {:.3f}, accuracy: {:.3f} ".format(epoch, result))
            
        
        
    def optimizers_dict(self, optimizer:str):
        self.optimizers = {'SGD': optim.SGD(self.net.parameters(), lr=0.001),
                           'SGD/momentum': optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
                           'Adam': optim.Adam(self.net.parameters())}
        return self.optimizers[optimizer]
    
    def resize_dict(self, inputs, resize_tp):
        m = nn.ZeroPad2d(80)
        self.resize = {'Interpolate': F.interpolate(inputs, size=(224, 224), mode='bicubic', align_corners=False),
                      'Padding': m(inputs)}
        return self.resize[resize_tp]
    
    def CNN_dict(self, net:str, nb_chn, nb_classes, nb_meta):
        self.nets = {'BasicCNN': CNNNetBasic(nb_chn,nb_classes),'BasicTabular':  TabularNetBasic(meta_size = nb_meta, num_classes = nb_classes),
                     'BasicCNNTabular': TabularCNNNetBasic(meta_size = nb_meta, img_size = nb_chn, num_classes = nb_classes),
                    'AlexNet': AlexNetResidual(nb_chn, nb_classes),
                    'ResNet18': resnet18maps(nb_chn,nb_classes), 'VGG16': VGG16Residual(nb_chn, nb_classes),
                    'DenseNet161': densenet161(input_sz = nb_chn, num_classes = nb_classes), 
                    'GoogleNet': GoogLeNet(googlenet(True, True, None,224, dropout_rate=0.2, num_classes=nb_classes)),
                    'SqueezeNet': squeezenet1_1(in_channels=nb_chn, num_classes=nb_classes)}
        
        return self.nets[net]
    
    def calculate_metrics(self, pred, target):
        """
        Compute different metrics to estimate the general error of th neural network error.

        :param pred : predicted output
        :param target : true labels
        :return : dictionnary, dictionnary of different metrics such as:
            - hamming          : compute average hamming loss
            - sample/precision : compute precision score for each instance and find their average 
            - samples/recall   : compute recall score for each instance and find their average 
            - samples/f1       : compute f1-score for each instance and find their average 
            - accuracy         : compute accuracy classification score for each instance and find their average 
        """

        return {'hamming': hamming_loss(target, pred),
                'samples/precision': precision_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
                'samples/recall': recall_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
                'samples/f1': f1_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
                'accuracy': accuracy_score(y_true = target, y_pred = pred)
                }