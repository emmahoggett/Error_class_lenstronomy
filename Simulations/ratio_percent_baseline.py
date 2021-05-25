import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from helpers.data_generation.file_management import read_hdf5
from helpers.data_generation.error_generation_chi2 import Residual, CombineDataset
from helpers.model.helpers_model import NeuralNet

import warnings
warnings.filterwarnings('ignore') 


# Build the four classes 
ratio_array = np.array([0.5, 0.75, 0.99])
percent_array = np.array([0.01, 0.02, 0.05])
size = 6000

batch_size = 50
max_epoch = 150

neuralnet_name = np.array(['BasicCNN', 'BasicTabular', 'BasicCNNTabular'])

for ratio in ratio_array:
    print("-----------Ratio: "+str(ratio)+" -----------\n")
    for percent in percent_array:
        print("----------------------- Percent: " + str(percent)+"-----------------------\n")
        
        res = Residual()
        res.build(size, ratio = ratio, per_error = percent*np.ones(3))

        str_ID =  "S"+str(size)+"R"+str(int(ratio*100))
        [final_array, metadata] = read_hdf5(str_ID, path = "data/dataset/")
        metadata ['ID'] = np.arange(0,final_array.shape[0])
        metadata = metadata.drop(columns=['percent', 'index'])
        data_set = CombineDataset(metadata,'ID','class',final_array)
        
        data_train, data_test = train_test_split(data_set,train_size=0.9,random_state=42)


        loader_train = DataLoader(data_train, batch_size = batch_size, 
                                  num_workers = 0, drop_last=True)

        loader_test = DataLoader(data_test, batch_size = batch_size, 
                                 num_workers = 0, drop_last=True)

        for netname in neuralnet_name:

            test_accSGD = np.zeros(max_epoch)
            netbasic = NeuralNet(netname, 'SGD/momentum')
            for epoch in range(max_epoch):
                netbasic.train(loader_train)
                res = netbasic.test(loader_test)
                test_accSGD[epoch] = res
            
            txt = "Finished Training: "+ netname +" - epoch: {:.3f} - accuracy: {:.3f} \n" 
            print(txt.format(netbasic.opti_epoch, netbasic.max_met))
