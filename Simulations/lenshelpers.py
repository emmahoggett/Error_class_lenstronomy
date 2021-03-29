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


    
class LensDataset:
    """Class that build the dataset in lenstronomy and """
    def __init__(self, size:int, errorID:int, seed = 0, center_x = 0, center_y = 0, param_range = None):
        """Initialization of the object LensDataset."""
        
        if param_range is None:
            amp_range = [20,24]; thetaE_range = [1, 2]; gamma_range = [1.8, 2.2]; q_range = [0.7, 1];
            phi_range = [0, math.pi]; Rsersic_range = [0.1,0.4]; nsersic_range = [0.8,5]; center_range = [-0.5,0.5]
            self.param_range = np.stack((thetaE_range,gamma_range,q_range,phi_range,amp_range, Rsersic_range, 
                                         nsersic_range,q_range,phi_range, center_range, center_range))
        elif type(param_range) is np.ndarray:
            if param_range.shape == (11,2):
                self.param_range = param_range
            else:
                raise Exception('param_range should be a (11,2) numpy array. The shape of param_range was: {}'.format(param_range.shape))
        else:
            raise Exception('param_range should be a numpy array. The type of param_range was: {}'.format(type(param_range)))
            
        self.center_x = center_x
        self.center_y = center_y
        
        psf_hdu = fits.open('data/psf.fits')
        kwargs_hst = HST().kwargs_single_band()
        kwargs_hst['kernel_point_source'] = np.array(psf_hdu[0].data)
        numPix = 64 

        kwargs_model = {'lens_model_list': ['PEMD'],        # list of lens models to be used
                        'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used
            }

        kwargs_numerics = {'point_source_supersampling_factor': 1}

        self.image_config = SimAPI(numpix=numPix, kwargs_single_band=kwargs_hst, kwargs_model=kwargs_model)
        self.img_sim = self.image_config.image_model_class(kwargs_numerics)
        
        self.size = size
        self.data = np.zeros((self.size,self.param_range.shape[0]))
        self.images = np.zeros((self.size, 1,64,64))
            
        for i in np.arange(0,self.param_range.shape[0]):
            self.data[:,i] = np.random.default_rng(seed).uniform(self.param_range[i,0],self.param_range[i,1],self.size)
            if i == 3 or i == 8:
                q =  np.random.default_rng(i).uniform(self.param_range[i,0],self.param_range[i,1],self.size)
                phi = np.random.default_rng(i).uniform(self.param_range[i,0],self.param_range[i,1],self.size)
                self.data[:,i-1] = (1-q[:])/(1+q[:])*np.cos(phi[:])
                self.data[:,i] = (1-q[:])/(1+q[:])*np.sin(phi[:])
                
            
        if errorID == 1:
            self.data[:,4:10] = self.data[0,4:10]
        elif errorID == 2:
            self.data[:,0:3] = self.data[0,0:3]
        
        
        self.metadata = pd.DataFrame(data=self.data)
        self.metadata.columns =  ['MASS_PROFILE_theta_E', 'MASS_PROFILE_gamma', 'MASS_PROFILE_e1', 'MASS_PROFILE_e2',
                                'SOURCE_PROFILE_amp', 'SOURCE_PROFILE_R_sersic', 'SOURCE_PROFILE_n_sersic', 
                                'SOURCE_PROFILE_e1', 'SOURCE_PROFILE_e2','SOURCE_PROFILE_center_x', 
                                'SOURCE_PROFILE_center_y']
        
        for i in np.arange(0,self.size):
            kwargs_spemd = {'theta_E': self.metadata['MASS_PROFILE_theta_E'][i], 'gamma': self.metadata['MASS_PROFILE_gamma'][i], 
                            'center_x': self.center_x, 'center_y': self.center_y, 
                            'e1':self.metadata['MASS_PROFILE_e1'][i], 'e2': self.metadata['MASS_PROFILE_e2'][i]}  
            kwargs_lens = [kwargs_spemd]
        
            # Sersic parameters in the initial simulation for the source
            kwargs_sersic = {'amp':  self.metadata['SOURCE_PROFILE_amp'][i], 'R_sersic': self.metadata['SOURCE_PROFILE_R_sersic'][i],
                             'n_sersic':  self.metadata['SOURCE_PROFILE_n_sersic'][i], 'e1':  self.metadata['SOURCE_PROFILE_e1'][i], 
                             'e2':  self.metadata['SOURCE_PROFILE_e2'][i], 'center_x': self.metadata['SOURCE_PROFILE_center_x'][i], 
                             'center_y':  self.metadata['SOURCE_PROFILE_center_y'][i]}
            kwargs_source = [kwargs_sersic]

            # generate image
            self.images[i,:,:] = self.img_sim.image(kwargs_lens, kwargs_source, kwargs_ps=None)
        
    
class  Residual:

    def __init__(self, size):
        """
        Args:
            path_config_model (string): 
        """
        random.seed(2)
        self.size = size

    def build(self, errorID, samedat_size = 4, path_data="data/dataSet/"):
        """
        This function builds the dataset by using the configuration file given by the user.
        This dataset ends up in the folder data/dataSet/. The file ID is 
        coded as following : [error number]_lens.h5 for residual maps and [error number]_meta.h5
            ex : E3S10 : 10th sample that correspond to a source and mass error(3).
        In the metadata, some informations are added such as the file ID and the type of error.
        !!! The errorID must absolutely match the configuration file or the data set will be false !!!
        Args:
            errorID (integer): Value that correspond to the type of error
                    - 0 : no error
                    - 1 : mass error
                    - 2 : source error
                    - 3 : source and mass error
            path_data (string): Location where the dataset will be saved. The default spot is 'data/dataSet/'.
        """
        # Use lenstronomy to make a data set
        dataset_model = LensDataset(size = self.size, errorID = errorID)
        self.path_data = path_data

        self.channels = dataset_model.images.shape[1]
        # Labeling 
        if errorID == 1:
            error = np.array([0,1,0])       #mass error
        elif errorID == 2:
            error = np.array([0,0,1])       #source error
        else:
            error = np.array([0,1,1])       #mass & source error
        
        metadata = pd.DataFrame()
        residuals = np.zeros((self.size*samedat_size, self.channels,64,64))
        k = 0
        
        for i in np.arange(0,self.size):
            metadata_temp = pd.concat([dataset_model.metadata.take([i])]*(samedat_size), ignore_index=True)
            bool_mdimg = np.array([], dtype='int')
            
            test = np.array([i])
            while test.shape[0]!=samedat_size:
                r=random.randint(1,self.size-1)
                if r not in test: test = np.append(test, r)
                
                
            for j in test:
                curr_error = np.array([0,0,0])
                if i!=j:
                    curr_error = error
                    
                    
                bool_mdimg = np.concatenate((bool_mdimg,curr_error))
                # Residual between two images i and j - the residual map is normalized and stored
                for i_ch in np.arange(0,self.channels):
                    image_model = dataset_model.images[j][i_ch]
                    image_real = dataset_model.images[i][i_ch] + dataset_model.image_config.noise_for_model(model = dataset_model.images[i][i_ch])
                    sigma = np.sqrt(dataset_model.img_sim.Data.C_D_model(model = dataset_model.images[i][i_ch]))
                    residuals[k,i_ch,:,:] = (image_real-image_model)/sigma
                k = k+1
                
            # Add the type of error in the metadata and the ID of the image
            metadata_temp['class'] = np.reshape(bool_mdimg, (-1, 3)).tolist()
            
            metadata = pd.concat([metadata,metadata_temp])
        # Store the data set as a hdf5 file
        store_hdf5(residuals, metadata, errorID, path = self.path_data)
    
