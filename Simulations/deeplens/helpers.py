#  this file contains helpers to handle binary files, combine and convert the dataset, comput the metrics and training/testing basic neural networks

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score,f1_score



def store_hdf5(images, labels, ID:str, path:str):
    """ Stores an array of images and the labels to HDF5 files.
    
        :param images   : np.array, residual maps (N, 1, 64, 64) to be stored
        :param labels   : pd.Dataframe, dataframe and labels (N, 11) to be stored
        :param ID       : string, ID of the file
        :param path     : string, path where the data is stored
    """

    # Create a new HDF5 file
    file = h5py.File(path+ID+"_lens.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset( "images", np.shape(images), h5py.h5t.IEEE_F64BE, data=images
    )
    file.close()

    labels.to_hdf(path +ID+'_meta.h5', "table")

def read_hdf5(ID_images:str, path):
    """ Reads images and metadatas from HDF5.
    
        :param ID_images : string, image ID
        :return images   : np.array, residual maps (N, 1, 64, 64) to be read
        :return labels   : pd.Dataframe, dataframe and labels (N, 11) to be read
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(path +ID_images+"_lens.h5", "r")

    images = np.array(file["/images"]).astype("float64")
    labels = pd.read_hdf(path +ID_images+'_meta.h5', "table")

    return images, labels
    


        
class CombineDataset(Dataset):
    """
    This class helps us to build a pytorch tensor by combining the images, images'
    metadata and labels.
    
    """

    def __init__(self, frame, id_col, label_name:str, image:str, nb_channel:int = 1):
        """
        
        :param frame      : pd.DataFrame, frame with the tabular data.
        :param id_col     : string, name of the column that connects image to tabular data
        :param label_name : string, name of the column with the label to be predicted
        :param path_imgs  : string, path to the folder where the images are.
        :param nb_channel : int, number of channels.
        """
        self.frame = frame
        self.id_col = id_col
        self.label_name = label_name
        self.image = image
        self.nb_channel = nb_channel
        
    def __len__(self):
        """
        
        :return : int, number of samples, which correspond to the length of the metadata.
        """
        return (self.frame.shape[0])

    def __getitem__(self, idx:int):
        """
        
        :param idx    : int, index of the desired image
        :return image : tensor, residual map or image
        :return feats : tensor, metadata of the residual map
        :return label : tensor, label of the residual map
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #complete image path and read
        img_name = self.frame[self.id_col].iloc[idx]
        image = self.image[img_name]
        image = torch.from_numpy(image.astype(np.float32))

        #get the other features to be used as training data
        feats = [feat for feat in self.frame.columns if feat not in [self.label_name,self.id_col]]
        feats  = np.array(self.frame[feats].iloc[idx])
        feats = torch.from_numpy(feats.astype(np.float32))
       
        
        #get label
        label = np.array(self.frame[self.label_name].iloc[idx])
        label = torch.from_numpy(label.astype(np.float32))

        return  image, feats, label


def calculate_metrics(pred, target):
    """
    Compute different metrics to estimate the general error of th neural network error.
    
    :param pred : predicted 
    :param target : 
    :return : dictionnary, dictionnary of different metrics such as:
        - hamming          : compute average hamming loss
        - sample/precision : compute precision score for each instance and find their average 
        - samples/recall   : compute recall score for each instance and find their average 
        - samples/f1       : compute f1-score for each instance and find their average 
        - samples/accuracy : compute accuracy classification score for each instance and find their average 
    """

    return {'hamming': hamming_loss(target, pred),
            'samples/precision': precision_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
            'samples/recall': recall_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
            'samples/f1': f1_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
            'accuracy': accuracy_score(y_true = target, y_pred = pred)
            }
            
            
def train_net(loader, net, optimizer, criterion, epoch:int):
    """
    Function to train a basic neural network on a given dataset (loader)
    
    :param loader    : CombineDataset, contain the combined residuals, metadata and labels
    :param net       : pytorch 2D convolutionnal neural network
    :param optimizer : pytorch optimizer
    :param criterion : pytorch loss function, in this problem a binary cross entropy must be used
    :param epoch     : int, epoch number
    :return          : pytorch 2D convolutionnal trained neural network
    """
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, meta_inputs,labels = data

        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        if net.typenet == 'conv':
            outputs = net(inputs)
        elif net.typenet == 'meta':
            outputs = net(meta_inputs)
        else :
            outputs = net(inputs, meta_inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    return net
                
def test_net(loader,net):
    """
    Make predictions on a given data-set(loader) with a 2D convolutionnal trained neural network(net).
    
    :param loader : CombineDataset, contain the combined residuals, metadata and labels
    :param net    : pytorch 2D convolutionnal neural network
    :return       : mean accuracy over the test dataset 
    """
    accuracy = 0
    iteration = 0
    with torch.no_grad():
        predictions = []
        targets = []
        for data in loader:
            images, meta_img, labels = data
            # forward + backward + optimize
            if net.typenet == 'conv':
                outputs = net(images)
            elif net.typenet == 'meta':
                outputs = net(meta_img)
            else :
                outputs = net(images, meta_img)

            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        result = calculate_metrics(np.round(np.array(predictions)), np.array(targets))

    return result['accuracy']
