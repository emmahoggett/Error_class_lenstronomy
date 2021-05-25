# this file contains the class that handle all neural networks and methods to train and test the performance

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from helpers.model.baseline import CNNNetBasic, TabularNetBasic, TabularCNNNetBasic
from helpers.model.densenet import densenet161, densenet121
from helpers.model.alexnet import AlexNet
from helpers.model.resnet18 import resnet18maps
from helpers.model.vgg import vgg11_bn, vgg16_bn
from helpers.model.googLeNet import googlenet, GoogLeNet
from helpers.model.squeezeNet import squeezenet1_1

from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class NeuralNet:
    """
    Class that contain all neural networks and methods to train and test the performance.
    The model is saved after each test with the neural network performance, while the optimal training is saved
    when performance are improved in the folder "data/model/checkpoints/".
    Thus, pickle files have the following structure :
        - Optimal training : [Net name]_optimal.pt
        - Current training : [Net name]_current.pt
    
    """
    def __init__(self, net:str, optimizer:str, int_channels:int = 1, out_channels:int = 2, meta_channels:int = 11):
        """
        
        :param net           : str, neural neutwork name
        :param optimizer     : str, optimizer's name
        :param in_channels   : int, image channel size - default : in_channels = 1
        :param out_channels  : int, number of output classes - default : out_channels = 2
        :param meta_channels : int, metadata parameters size - default : meta_channels = 11
        """
        self.net_name = net
        self.net = self.CNN_dict(net, int_channels, out_channels, meta_channels)
        self.max_met = 0
        self.current_epoch  = 0
        self.criterion = nn.BCELoss()
        self.optimizer = self.optimizers_dict(optimizer)
        self.epoch_metric = []
        self.save_path = "data/model/checkpoints/"+self.net_name
        
    
    def train (self, loader_train, resize_tp:str = 'Padding')->None:
        """
        
        :param loader_train : DataLoader, training set with images (1,64,64), metadata and labels.
        :param resize_tp    : str, resizing method name - default : resize_tp = 'Padding'
        """
        self.resize_tp = resize_tp
        self.current_epoch +=1
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
            
            
    def test (self, loader_test, metric:str = 'auc', verbose:bool = True)->float:
        """
        
        :param loader_test : DataLoader, test set with images (1,64,64), metadata and labels.
        :param metric      : str, metric name that optimize our neural network
        :param verbose     : bool, show the epoch and performance improvements if true
        :return            : float, current performance of the neural network for the given metric
        """
        self.metric = metric
        
        with torch.no_grad():
            predictions = []
            targets = []
            for data in loader_test:
                images, meta, labels = data
                # resizing
                if self.net_name!='BasicCNN' and self.net_name!='BasicTabular' and  self.net_name!='BasicCNNTabular':
                    images = self.resize_dict(images, self.resize_tp)
                # predictions
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
        self.epoch_metric.append(result[metric])
        
        # update section
        if len(self.epoch_metric)==1 and self.metric == 'hamming':
            self.max_met = 1
        
        self._update_(result[metric], verbose)
        return result[metric]
    
    
    def _update_(self, result:float, verbose:bool)->None:
        """
        
        :param result  : float, current performance of the neural network for the given metric
        :param verbose : bool, show the epoch and performance improvements if true
        """
        
        self.save_checkpoint('_current')
        bool_ham = self.metric == 'hamming'
        if result > self.max_met and self.metric != 'hamming':
            self.max_met = result
            self.save_checkpoint('_optimal')
            if verbose:
                txt = "epoch: {:.3f}, "+self.metric+": {:.3f}" 
                print(txt.format(self.current_epoch, result))
                
        elif self.metric == 'hamming' and result < self.max_met:
            self.max_met = result
            self.save_checkpoint('_optimal')
            if verbose:
                txt = "epoch: {:.3f}, "+self.metric+": {:.3f}" 
                print(txt.format(self.current_epoch, result))
            
            
        
    def optimizers_dict(self, optimizer:str):
        """
        
        :param optimizer : str, optimizer's name
        :return          : optim, neural network optimizer
        """
        optimizers = {'SGD': optim.SGD(self.net.parameters(), lr=0.001),
                      'SGD/momentum': optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
                      'Adam': optim.Adam(self.net.parameters(), eps = 0.07)}
        return optimizers[optimizer]
    
    
    def resize_dict(self, inputs:torch.Tensor, resize_tp:str)->torch.Tensor:
        """
        
        :param inputs    : torch.Tensor, image of size (1,64,64)
        :param resize_tp : str, resize method's name
        :return          : torch.Tensor, resized image of size (1,224,224)
        """
        m = nn.ZeroPad2d(80)
        interp = F.interpolate(inputs, size=(224, 224), mode='bicubic', align_corners=False)
        resize = {'Interpolate': interp,
                  'Padding': m(inputs)}
        return resize[resize_tp]
    
    
    def CNN_dict(self, net:str, in_channels:int, out_channels:int, meta_channels:int):
        """
        
        :param net           : str, neural neutwork name
        :param in_channels   : int, image channel size 
        :param out_channels  : int, number of output classes
        :param meta_channels : int, metadata parameters size
        :return              : nn.Module, neural network model
        """
        nets = {'BasicCNN': CNNNetBasic(in_channels,out_channels),
                'BasicTabular':  TabularNetBasic(meta_channels = meta_channels, out_channels = out_channels),
                'BasicCNNTabular': TabularCNNNetBasic(meta_channels = meta_channels, in_channels = in_channels, out_channels = out_channels),
                'AlexNet': AlexNet(in_channels, out_channels),
                'ResNet18': resnet18(in_channels,out_channels), 
                'VGG11': vgg11_bn(in_channels = in_channels, out_channels = out_channels),
                'VGG16': vgg16_bn(in_channels = in_channels, out_channels = out_channels),
                'DenseNet161': densenet161(in_channels = in_channels, out_channels = out_channels),
                'DenseNet121': densenet121(in_channels = in_channels, out_channels = out_channels),
                'GoogleNet': GoogLeNet(googlenet(True, None, 224, in_channels, out_channels)),
                'SqueezeNet': squeezenet1_1(in_channels = in_channels, out_channels = out_channels)}
        return nets[net]
    
    
    def calculate_metrics(self, pred:np.array, target:np.array):
        """
        Compute different metrics to estimate the general error of th neural network error.

        :param pred   : np.array, predicted output
        :param target : np.array, true labels
        :return       : dictionnary, dictionnary of different metrics such as:
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
    
    def save_checkpoint(self, name:str)->None:
        """
        
        Save a checkpoint file with model, optimizer, maximum performance and performance over epoch.
        
        :param name : str, name of the saved file such that "data/model/checkpoints/[Net name][name].pt"
        """
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'metric': self.max_met,
            'epoch/metric': np.array(self.epoch_metric)
        }, self.save_path + name + ".pt")
        

    def load_checkpoint(self, name:str)->None:
        """
        
        Load a checkpoint for a given neural network, with model, optimizer, maximum performance and performance over epoch.
        
        :param name : str, name of the loaded file such that "data/model/checkpoints/[Net name][name].pt"
        """
        checkpoint = torch.load(self.save_path + name + ".pt")
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.eval()
        self.current_epoch = checkpoint['epoch']
        self.max_met = checkpoint['metric']
        self.epoch_metric = checkpoint['epoch/metric'].tolist()