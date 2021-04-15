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

from errors import MassError, SourceError

    
class LensDataset:
    """Class that build residual and metadata in lenstronomy. This class """
    def __init__(self, size:int, seed:int = 0, center_x:float = 0, center_y:float = 0, mass_range = None, source_range = None):
        """Initialization of the object LensDataset.
        
        
        :param size: dataset size
        :param seed: random seed for the uniform distribution
        :param center_x: coordinate center of the lens mass profile on the x-axis
        :param center_y: coordinate center of the lens mass profile on the y-axis
        :param param_range: 11x2 matrix, with specified range of the mass and source light profile"""

        
        psf_hdu = fits.open('data/psf.fits')
        kwargs_hst = HST().kwargs_single_band()
        kwargs_hst['kernel_point_source'] = np.array(psf_hdu[0].data)
        numPix = 64 

        kwargs_model = {'lens_model_list': ['PEMD'],        # list of lens models to be used
                        'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used
            }
        self.size = size
        self.image_config = SimAPI(numpix=numPix, kwargs_single_band=kwargs_hst, kwargs_model=kwargs_model)
        self.img_sim = self.image_config.image_model_class()
        
        
        # Build dictionnary
        self.mass_error = MassError(size,seed, center_x, center_y, mass_range)
        self.source_error = SourceError(size,seed,source_range)
        
        self.metadata = pd.concat([self.mass_error.metadata,self.source_error.metadata], axis =1)
        self.images = np.zeros((self.size, 1,64,64))
        
        for i in np.arange(0,self.size): 
            kwargs_lens = self.mass_error.get_kwargs(i)
            kwargs_source = self.source_error.get_kwargs(i)

            # generate image
            self.images[i,:,:] = self.img_sim.image(kwargs_lens, kwargs_source, kwargs_ps=None)
        
    
class  Residual:

    def build(self, size, errorID, ratio = 0.75, per_error = 0.01, num_class = 3, center_x:float = 0, center_y:float = 0,
              mass_range = None, source_range = None, path_data="data/dataSet/"):
        """
        This function builds the dataset by using the configuration file given by the user.
        This dataset ends up in the folder data/dataSet/. The file ID is 
        coded as following : [error number]_lens.h5 for residual maps and [error number]_meta.h5
            ex : E3S10 : 10th sample that correspond to a source and mass error(3).
        In the metadata, some informations are added such as the file ID and the type of error.
        !!! The errorID must absolutely match the configuration file or the data set will be false !!!
        Args:
            errorID (integer): Value that correspond to the type of error
                    - 1 : mass error
                    - 2 : source error
                    - 3 : source and mass error
            path_data (string): Location where the dataset will be saved. The default spot is 'data/dataSet/'.
        """
        # Use lenstronomy to make a data set
        
        self.size = round(size/num_class)
        dataset_model = LensDataset(size = size, center_x = center_x, center_y = center_y, mass_range = mass_range, source_range= source_range)
        self.path_data = path_data

        self.channels = dataset_model.images.shape[1]
        metadata = pd.DataFrame()
        residuals = np.zeros((self.size, self.channels,64,64))
        bool_mdimg = np.array([], dtype='int')
        k = 0
        
        true_idx = np.array([], dtype='int')
        np.random.seed(errorID*10)
        while true_idx.shape[0]<((1-ratio)*self.size):
            r=np.random.randint(0,size-1)
            if r not in true_idx: true_idx = np.append(true_idx, r)
                
        for i in true_idx:
            metadata_temp = dataset_model.metadata.take([i])
            
            img_test = np.zeros((1, self.channels,64,64))
            
            for i_ch in np.arange(0,self.channels):
                image_model = dataset_model.images[i][i_ch]
                image_real = dataset_model.images[i][i_ch] + dataset_model.image_config.noise_for_model(model = dataset_model.images[i][i_ch])
                sigma = np.sqrt(dataset_model.img_sim.Data.C_D_model(model = dataset_model.images[i][i_ch]))
                residuals[k,i_ch,:,:] = (image_real-image_model)/sigma
            bool_mdimg = np.concatenate((bool_mdimg, np.array([1,0,0])))
            metadata = pd.concat([metadata,dataset_model.metadata.take([i])])
            k = k+1
            
        false_idx = np.array([], dtype='int')
        np.random.seed(errorID*10)
        while false_idx.shape[0]<(self.size-true_idx.shape[0]):
            r=np.random.randint(0,size-1)
            if r not in false_idx: false_idx = np.append(false_idx, r)

              
        img_test = np.zeros((self.channels,64,64))        
        for i in false_idx:
            kwargs_lens = dataset_model.mass_error.add_error(i,per_error)
            kwargs_source = dataset_model.source_error.add_error(i,per_error)
            error = dataset_model.mass_error.type_error + dataset_model.source_error.type_error
            if errorID == 1:
                # Sersic parameters in the initial simulation for the source
                kwargs_source = dataset_model.source_error.get_kwargs(i)
                error = dataset_model.mass_error.type_error
            elif errorID == 2:
                kwargs_lens = dataset_model.mass_error.get_kwargs(i)
                error = dataset_model.source_error.type_error

            # generate image
            img_test = dataset_model.img_sim.image(kwargs_lens, kwargs_source, kwargs_ps=None)
            
            for i_ch in np.arange(0,self.channels):
                image_model = dataset_model.images[i][i_ch]
                image_real = img_test + dataset_model.image_config.noise_for_model(model = img_test)
                sigma = np.sqrt(dataset_model.img_sim.Data.C_D_model(model = img_test))
                residuals[k,i_ch,:,:] = (image_real-image_model)/sigma
            bool_mdimg = np.concatenate((bool_mdimg, error))
            metadata = pd.concat([metadata,dataset_model.metadata.take([i])])
            k = k+1
            
        metadata['class'] = np.reshape(bool_mdimg, (-1, 3)).tolist()
        self.residuals = residuals
        self.metadata = metadata
        # Store the data set as a hdf5 file
        store_hdf5(residuals, metadata, errorID, path = self.path_data)
