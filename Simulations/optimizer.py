import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from helpers import read_hdf5, CombineDataset
from lenshelpers import Residual

from model.helpers_model import NeuralNet

import warnings
warnings.filterwarnings('ignore') 


# Build the four classes 
config_repo_model = 'data/configFile/config_model'
ratio = 0.75
percent = 0.01
size = 6000

batch_size = 50
max_epoch = 50

opti_name = np.array(['SGD', 'SGD/momentum', 'Adam'])



for i in np.arange(1,4):
    #model_name = config_repo_model + str(i) + '.yaml'
    res = Residual()
    res.build(size, i, ratio = ratio, per_error = percent)

metadata = pd.DataFrame()

for i in np.arange(1,4):
    str_ID = "E"+str(i)+"P"+str(int(percent*100))+"R"+str(int(ratio*100))
    [img, meta] = read_hdf5(str_ID, path = "data/dataSet/")
    metadata = pd.concat([metadata,meta], ignore_index=True)
    if i == 1:
        final_array = img
    else:
         final_array = np.concatenate((final_array, img))
metadata ['ID'] = np.arange(0,final_array.shape[0])
data_set = CombineDataset(metadata,'ID','class',final_array)

data_train, data_test = train_test_split(data_set,train_size=0.9,random_state=42)


loader_train = DataLoader(data_train, batch_size = batch_size, 
                          num_workers = 0, drop_last=True)

loader_test = DataLoader(data_test, batch_size = batch_size, 
                         num_workers = 0, drop_last=True)

normalize = np.array([np.mean(final_array), np.std(final_array)])
plt.figure()
for opti in opti_name:
    print('----------------------------------------------------------------------------------')

    test_SGDnorm = np.zeros(max_epoch)
    netbasic = NeuralNet('BasicCNN', opti)
    for epoch in range(max_epoch):
        netbasic.train(loader_train, normalize = normalize)
        res = netbasic.test(loader_test,epoch)
        test_SGDnorm[epoch] = res

    txt = "Finished Training- Normalize: "+ opti +" - epoch: {:.3f} - auc: {:.3f} \n" 
    print(txt.format(netbasic.opti_epoch, netbasic.max_met))
    
    test_SGD = np.zeros(max_epoch)
    netbasic = NeuralNet('BasicCNN', opti)
    for epoch in range(max_epoch):
        netbasic.train(loader_train)
        res = netbasic.test(loader_test,epoch)
        test_SGD[epoch] = res

    txt = "Finished Training - Not normalize: "+ opti +" - epoch: {:.3f} - auc: {:.3f} \n" 
    print(txt.format(netbasic.opti_epoch, netbasic.max_met))
    
    plt.plot(np.arange(1, max_epoch+1), test_SGDnorm, label='Norm '+opti)
    plt.plot(np.arange(1, max_epoch+1), test_SGD, label=opti)

plt.savefig('figures/optimizers.jpeg')
plt.show()