import numpy as np
import pandas as pd
import random
import os
import deeplenstronomy.deeplenstronomy as dl
import torch

from numpy import genfromtxt
from torch.utils.data import Dataset



class  Residual:

    def __init__(self, path_config_img, path_config_model, size):
        """
        Args:
            path_config_img (string): Configuration path for the images
            path_config_model (string): 
            size (integer): Size of the data set
        """
        random.seed(2)
        self.path_config_img = path_config_img
        self.path_config_model = path_config_model
        self.size = size

    def build(self, errorID):
        """
        This function builds the dataset by using the configuration file given by the user.
        This dataset ends up in the folder DeeplenstronomyDataset/DataSet/. The file ID is 
        coded as following : E[error number]S[sample number].csv
            ex : E3S10 : 10th sample that correspond to a source and mass error(3).
        In the metadata, some informations are added such as the file ID and the type of error.
        !!! The errorID must absolutely match the configuration file or the data set will be false !!!
        Args:
            errorID (integer): Value that correspond to the type of error
                    - 0 : no error
                    - 1 : mass error
                    - 2 : source error
                    - 3 : source and mass error
        """
        # Use deeplenstronomy to make a data set
        dataset_model = dl.make_dataset(self.path_config_model)
        dataset_img = dl.make_dataset(self.path_config_img)
        
        metadata = pd.DataFrame()
        k = 0
        for i in np.arange(1,self.size):
            metadata_source_mass = pd.concat([dataset_model.CONFIGURATION_1_metadata.take([i])]*(4), 
                                             ignore_index=True)
            bool_mdimg = []
            ID_img = []
            
            test = np.array([i])
            while test.shape[0]!=4:
                r=random.randint(1,self.size-1)
                if r not in test: test = np.append(test, r)
                 
            for j in test:
                if i!=j:
                    bool_mdimg.append(errorID)
                    file_name = "DeeplenstronomyDataset/DataSet/"+"E"+str(errorID)+"S"+str(k)+".csv"
                    ID_img.append('E'+str(errorID) + 'S' + str(k))
                else:
                    bool_mdimg.append(0)
                    file_name = "DeeplenstronomyDataset/DataSet/"+"E"+str(0)+"S"+str(k)+".csv"
                    ID_img.append('E'+str(0) + 'S' + str(k))
                
                # Residual between two images i and j
                residual = dataset_img.CONFIGURATION_1_images[j][0]-dataset_model.CONFIGURATION_1_images[i][0]
                file_array = residual.ravel()
                np.savetxt(file_name, file_array, delimiter=",")
                k = k + 1
            # Add the type of error in the metadata and the ID of the image
            metadata_source_mass['class'] = bool_mdimg
            metadata_source_mass['ID'] = ID_img
            metadata = pd.concat([metadata,metadata_source_mass])
        file_name = "DeeplenstronomyDataset/DataSet/"+"MetaE"+str(errorID)+".csv"
        metadata.to_csv(file_name,index=False)

        
class CombineDataset(Dataset):
    """
    This class helps us to build a pytorch tensor by combining the images and the
    metadata.
    """

    def __init__(self, frame, id_col, label_name, path_imgs, nb_channel = 1):
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
        self.path_imgs = path_imgs
        self.nb_channel = nb_channel

    def __len__(self):
        return (self.frame.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #complete image path and read
        img_name = self.frame[self.id_col].iloc[idx]
        path = os.path.join(self.path_imgs,img_name) + '.csv'
        image = genfromtxt(path, delimiter=',')
        image = image.reshape(self.nb_channel, 64, 64)
        image = torch.from_numpy(image.astype(np.float32))

        #get the other features to be used as training data
        feats = [feat for feat in self.frame.columns if feat not in [self.label_name,self.id_col]]
        feats  = np.array(self.frame[feats].iloc[idx])
        feats = torch.from_numpy(feats.astype(np.float32))
       
        
        #get label
        label = np.array(self.frame[self.label_name].iloc[idx])
        label = torch.from_numpy(label.astype(np.float32))

        return image, feats, label
