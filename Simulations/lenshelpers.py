import numpy as np
import pandas as pd
import random
import os

from lenstronomy.LensModel.Profiles.pemd import PEMD
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from astropy.io import fits
from lenstronomy.SimulationAPI.sim_api import SimAPI

from helpers import*
import math

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.

    
class LensDataset():
    def __init__(self, size, errorID, seed = 42, center_x = 0, center_y = 0, amp_range = [20,24], thetaE_range = [1, 2], 
                gamma_range = [1.8, 2.2], q_range = [0.7, 1], phi_range = [0, math.pi], Rsersic_range = [0.1,0.4], nsersic_range = [0.8,5], center_range = [-0.5,0.5]):
        self.size = size
        self.param_range = np.stack((thetaE_range,gamma_range,q_range,phi_range,amp_range, Rsersic_range, nsersic_range,q_range,phi_range, center_range, center_range))
        self.data = np.zeros((self.size,self.param_range.shape[0]))
        self.seed = seed
        self.center_x = center_x
        self.center_y = center_y
        self.image = np.zeros((self.size, 1,64,64))
        psf_hdu = fits.open('data/psf.fits')
        kwargs_hst = HST().kwargs_single_band()
        kwargs_hst['kernel_point_source'] = np.array(psf_hdu[0].data)
        numPix = 64 

        kwargs_model = {'lens_model_list': ['PEMD'],  # list of lens models to be used
                        'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used
            }

        kwargs_numerics = {'point_source_supersampling_factor': 1}

        self.sim_hst = SimAPI(numpix=numPix, kwargs_single_band=kwargs_hst, kwargs_model=kwargs_model)
        self.imSim_hst = self.sim_hst.image_model_class(kwargs_numerics)
        
        
        for i in np.arange(0,self.param_range.shape[0]):
            self.data[:,i] = np.random.default_rng(self.seed).uniform(self.param_range[i,0],self.param_range[i,1],self.size)
            
        for i in np.array([2, 7]):
            q =  np.random.default_rng(i).uniform(self.param_range[i,0],self.param_range[i,1],self.size)
            phi = np.random.default_rng(i).uniform(self.param_range[i,0],self.param_range[i,1],self.size)
            self.data[:,i] = (1-q[:])/(1+q[:])*np.cos(phi[:])
            self.data[:,i+1] = (1-q[:])/(1+q[:])*np.sin(phi[:])
        
        self.data[:,9] = np.random.default_rng(self.seed+1).uniform(self.param_range[9,0],self.param_range[9,1],self.size)
        
        self.params = pd.DataFrame(data=self.data)
        self.params.columns =  ['MASS_PROFILE_theta_E', 'MASS_PROFILE_gamma', 'MASS_PROFILE_e1', 'MASS_PROFILE_e2',
                                'SOURCE_PROFILE_amp', 'SOURCE_PROFILE_R_sersic', 'SOURCE_PROFILE_n_sersic', 'SOURCE_PROFILE_e1', 
                                'SOURCE_PROFILE_e2','SOURCE_PROFILE_center_x', 'SOURCE_PROFILE_center_y']
        random.seed(self.seed)
        if errorID == 1:
            for i in np.arange(4,self.param_range.shape[0]):
                self.data[:,i] = (self.param_range[i,0] + random.random()*(self.param_range[i,1]-self.param_range[i,0]))*np.ones((self.size))
        elif errorID == 2:
            for i in np.arange(0,4):
                self.data[:,i] = (self.param_range[i,0] + random.random()*(self.param_range[i,1]-self.param_range[i,0]))*np.ones((self.size))
        
        
        for i in np.arange(0,self.size):
            kwargs_spemd = {'theta_E': self.params['MASS_PROFILE_theta_E'][i], 'gamma': self.params['MASS_PROFILE_gamma'][i], 'center_x': self.center_x, 'center_y': self.center_y, 
                            'e1':self.params['MASS_PROFILE_e1'][i], 'e2': self.params['MASS_PROFILE_e2'][i]}  
            kwargs_lens = [kwargs_spemd]
        
            # Sersic parameters in the initial simulation for the source
            kwargs_sersic = {'amp':  self.params['SOURCE_PROFILE_amp'][i], 'R_sersic': self.params['SOURCE_PROFILE_R_sersic'][i], 'n_sersic':  self.params['SOURCE_PROFILE_n_sersic'][i], 
                             'e1':  self.params['SOURCE_PROFILE_e1'][i], 'e2':  self.params['SOURCE_PROFILE_e2'][i], 'center_x': self.params['SOURCE_PROFILE_center_x'][i], 
                             'center_y':  self.params['SOURCE_PROFILE_center_y'][i]}
            kwargs_source = [kwargs_sersic]

            # generate image
            self.image[i,:,:] = self.imSim_hst.image(kwargs_lens, kwargs_source, kwargs_ps=None)
            
    def metadata(self):
        return self.params
    def images(self):
        return self.image
    def image_config(self):
        return self.sim_hst
        
    
class  Residual:

    def __init__(self, size):
        """
        Args:
            path_config_model (string): 
        """
        random.seed(2)
        self.size = size

    def build(self, errorID, path_data="data/dataSet/"):
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
        dataset_model = LensDataset(self.size, errorID)
        self.path_data = path_data

        self.channels = dataset_model.images().shape[1]

        if errorID == 1:
            error = np.array([0,1,0])
        elif errorID == 2:
            error = np.array([0,0,1])
        else:
            error = np.array([0,1,1])
        
        metadata = pd.DataFrame()
        residuals = np.zeros((self.size*4, self.channels,64,64))
        k = 0
        
        for i in np.arange(0,self.size):
            metadata_temp = pd.concat([dataset_model.metadata().take([i])]*(4), ignore_index=True)
            bool_mdimg = np.array([], dtype='int')
            
            test = np.array([i])
            while test.shape[0]!=4:
                r=random.randint(1,self.size-1)
                if r not in test: test = np.append(test, r)
                
                
            for j in test:
                curr_error = np.array([0,0,0])
                if i!=j:
                    curr_error = error
                    
                    
                bool_mdimg = np.concatenate((bool_mdimg,curr_error))
                # Residual between two images i and j
                for i_ch in np.arange(0,self.channels):
                    image_model = dataset_model.images()[j][i_ch]
                    image_real = dataset_model.images()[i][i_ch] + dataset_model.image_config().noise_for_model(model=dataset_model.images()[i][i_ch])
                    residuals[k,i_ch,:,:] =  image_real-image_model
                k = k+1
            # Add the type of error in the metadata and the ID of the image
            metadata_temp['class'] =np.reshape(bool_mdimg, (-1, 3)).tolist()
            
            metadata = pd.concat([metadata,metadata_temp])
        print(residuals.shape)
        store_hdf5(residuals, metadata, errorID, path = self.path_data)
    
