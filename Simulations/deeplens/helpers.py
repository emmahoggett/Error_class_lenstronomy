import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score,f1_score
# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.

hdf5_dir = "data/dataSet/"


def store_hdf5(images, labels, ID, path = hdf5_dir):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 64, 64, 1) to be stored
        labels       labels array, (N, 1) to be stored
    """

    # Create a new HDF5 file
    file = h5py.File(path +str(ID)+"_lens.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.IEEE_F64BE, data=images
    )
    file.close()

    labels.to_hdf(path +str(ID)+'_meta.h5', "table")

def read_hdf5(ID_images, path = hdf5_dir):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 1, 64, 64) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(path +str(ID_images)+"_lens.h5", "r")

    images = np.array(file["/images"]).astype("float64")
    labels = pd.read_hdf(path +str(ID_images)+'_meta.h5', "table")

    return images, labels
    


        
class CombineDataset(Dataset):
    """
    This class helps us to build a pytorch tensor by combining the images and the
    metadata.
    """

    def __init__(self, frame, id_col, label_name, image, nb_channel = 1):
        """
        Args:
            frame (pd.DataFrame): Frame with the tabular data.
            id_col (string): Name of the column that connects image to tabular data
            label_name (string): Name of the column with the label to be predicted
            path_imgs (string): Path to the folder where the images are.
            nb_channel (int): Number of channels.
        """
        self.frame = frame
        self.id_col = id_col
        self.label_name = label_name
        self.image = image
        self.nb_channel = nb_channel
        
    def __len__(self):
        """Return the number of samples, which correspond to the length of the metadata."""
        return (self.frame.shape[0])

    def __getitem__(self, idx):
        """"""
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


def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    return {'match/ratio': accuracy_score(target, pred, normalize = True, sample_weight = None), 
            'hamming': hamming_loss(target, pred),
            'samples/precision': precision_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
            'samples/recall': recall_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
            'samples/f1': f1_score(y_true = target, y_pred = pred, average = 'samples', zero_division = 1),
            'accuracy': accuracy_score(y_true = target, y_pred = pred)
            }
            
            
def train_net(loader, net, optimizer, criterion, epoch):
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
def test_net(loader,net):
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
                accuracy+=result['samples/recall']
                iteration+=1

        mean_accuracy =accuracy/(iteration+1)
        return mean_accuracy
