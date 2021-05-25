#  this file contains helpers to handle binary files, combine and convert the dataset, comput the metrics and training/testing basic neural networks

import numpy as np
import pandas as pd
import h5py



def store_hdf5(images, labels, ID:str, path:str = "data/dataset/"):
    """ 
    Stores an array of images and the labels to HDF5 files.
    
    :param images   : np.array(N, 1, 64, 64), residual maps to be stored
    :param labels   : pd.Dataframe(N, 11), dataframe and labels to be stored
    :param ID       : string, ID of the file
    :param path     : string, path where the data is stored - default : path = "data/dataset/"
    """

    #create a new HDF5 file
    file = h5py.File(path+ID+"_lens.h5", "w")

    #create a dataset in the file
    dataset = file.create_dataset( "images", np.shape(images), h5py.h5t.IEEE_F64BE, data=images)
    file.close()

    labels.to_hdf(path +ID+'_meta.h5', "table")
    
    

def read_hdf5(ID_images:str, path:str = "data/dataset/"):
    """ 
    Reads images and metadatas from HDF5.
    
    :param ID_images : string, image ID
    :param path      : string, path of the dataset - default : path = "data/dataset/"
    :return images   : np.array, residual maps (N, 1, 64, 64) to be read
    :return labels   : pd.Dataframe, dataframe and labels (N, 11) to be read
    """
    images, labels = [], []

    #open the HDF5 file
    file = h5py.File(path +ID_images+"_lens.h5", "r")

    images = np.array(file["/images"]).astype("float64")
    labels = pd.read_hdf(path +ID_images+'_meta.h5', "table")

    return images, labels