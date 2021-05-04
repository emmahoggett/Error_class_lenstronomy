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
from model.alexnet import AlexNet
from model.resnet18 import resnet18maps
from model.vgg16 import VGG16
from model.googLeNet import googlenet, GoogLeNet
from model.squeezeNet import squeezenet1_1

from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class NeuralNet:
    def __init__(self, net:str, optimizer:str, nb_chn:int = 1, nb_classes:int = 2, meta_in:int = 11):
        self.net_name = net
        self.net = self.CNN_dict(net, nb_chn, nb_classes, meta_in)
        self.max_met = 0
        self.criterion = nn.BCELoss()
        self.optimizer = self.optimizers_dict(optimizer)
        self.epoch_metric = []
        self.current_epoch = 0
        self.save_path = "model/checkpoints/"+self.net_name + ".pt"
        
    
    def train (self, loader_train, resize_tp:str = 'Padding'):
        self.resize_tp = resize_tp
        self.current_epoch = self.current_epoch + 1
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
            
            
    def test (self, loader_test, metric:str = 'auc'):
        self.metric = metric
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
        self._update_(result[metric])
        self.epoch_metric.append(result[metric])
        return result[metric]
    
    
    def _update_(self, result):
        if result > self.max_met:
            self.max_met = result
            self.opti_epoch =  self.current_epoch
            self.save_checkpoint()
            txt = "epoch: {:.3f}, "+self.metric+": {:.3f}" 
            print(txt.format(self.current_epoch, result))
            
        
    def optimizers_dict(self, optimizer:str):
        self.optimizers = {'SGD': optim.SGD(self.net.parameters(), lr=0.001),
                           'SGD/momentum': optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
                           'Adam': optim.Adam(self.net.parameters(), eps = 0.1)}
        return self.optimizers[optimizer]
    
    
    def resize_dict(self, inputs, resize_tp):
        m = nn.ZeroPad2d(80)
        self.resize = {'Interpolate': F.interpolate(inputs, size=(224, 224), mode='bicubic', align_corners=False),
                       'Padding': m(inputs)}
        return self.resize[resize_tp]
    
    
    def CNN_dict(self, net:str, nb_chn, nb_classes, nb_meta):
        self.nets = {'BasicCNN': CNNNetBasic(nb_chn,nb_classes),'BasicTabular':  TabularNetBasic(meta_size = nb_meta, num_classes = nb_classes),
                     'BasicCNNTabular': TabularCNNNetBasic(meta_size = nb_meta, img_size = nb_chn, num_classes = nb_classes),
                     'AlexNet': AlexNet(nb_chn, nb_classes),
                     'ResNet18': resnet18maps(nb_chn,nb_classes), 'VGG16': VGG16(nb_chn, nb_classes),
                     'DenseNet161': densenet161(input_sz = nb_chn, num_classes = nb_classes), 
                     'GoogleNet': GoogLeNet(googlenet(True, True, None, 224, nb_chn, nb_classes)),
                     'SqueezeNet': squeezenet1_1(in_channels=nb_chn, num_classes=nb_classes)}
        return self.nets[net]
    
    
    def calculate_metrics(self, pred, target):
        """
        Compute different metrics to estimate the general error of th neural network error.

        :param pred : predicted output
        :param target : true labels
        :return : dictionnary, dictionnary of different metrics such as:
            - hamming          : compute average hamming loss
            - precision        : compute precision score for each instance and find their average 
            - recall           : compute recall score for each instance and find their average 
            - f1               : compute f1-score for each instance and find their average 
            - exactmatch       : compute exact match ratio, such that partially correct predictions are considered as false
            - accuracy         : compute accuracy classification score for each instance and find their average 
        """
        
        accuracy = 0
        auc = 0
        for i in range(0, pred.shape[1]):
            accuracy = accuracy + accuracy_score(y_true = target[:,i], y_pred = pred[:,i])
            auc = auc + roc_auc_score(target[:,i], pred[:,i], average = 'samples')  
        return {'hamming': hamming_loss(target, pred),
                'precision': precision_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
                'recall': recall_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
                'f1': f1_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
                'exactmatch': accuracy_score(y_true = target, y_pred = pred),
                'accuracy' : accuracy/pred.shape[1],
                'auc': auc/pred.shape[1]
                }
    
    def save_checkpoint(self):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.opti_epoch,
            'metric': self.max_met,
            'epoch/metric': np.array(self.epoch_metric)
        }, self.save_path)
        

    def load_checkpoint(self):
        checkpoint = torch.load(self.save_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.opti_epoch  = checkpoint['epoch']
        self.current_epoch = checkpoint['epoch']
        self.max_met = checkpoint['metric']
        self.epoch_metric = checkpoint['epoch/metric'].tolist()