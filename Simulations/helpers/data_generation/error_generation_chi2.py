#  this file contains classes that build the residual maps with lenstronomy

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations
import scipy.stats as st
from math import log
from statistics import mean

from lenstronomy.LensModel.Profiles.pemd import PEMD
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from astropy.io import fits
from lenstronomy.SimulationAPI.sim_api import SimAPI

from helpers.data_generation.file_management import store_hdf5
from helpers.data_generation.errors import LensMassError, SourceError

import torch
from torch.utils.data import Dataset

    
class LensDataset:
    """
    Class that build residual and metadata in lenstronomy. This class configure the observation setting and use a psf file in the
    Time Delay Lens Modeling Challenge in the Rung0 training exercise.
        path : data/rung0/code1/f160w-seed3/drizzled_image/psf.fits
    
    Data is generated over the HST model with kwargs model such that:
            - Lens Mass model : PEMD
            - Source model    : SERSIC_ELLIPSE
            
    Note : To add additionnal error the kwargs model, the image configuration and the image must be redefined.
    """
    def __init__(self, size:int, percent:float, seed:int = 0, numPix:int = 64, center_x:float = 0, center_y:float = 0, 
                 mass_range = None, source_range = None)->None:
        """
        Initialization of the object LensDataset.
        
        
        :param size       : int, dataset size
        :param percent    : float, percentage of error for the mass lens and the source error
        :param seed       : int, random seed to determine if its a negative or a positive error - default : seed = 0
        :param numPix     : int, size of the image, it must be a power of 2 - default : numPix = 64
        :param center_x   : coordinate center of the lens mass profile on the x-axis - default : center_x = 0
        :param center_y   : coordinate center of the lens mass profile on the y-axis - default : center_y = 0
        :param mass_range : np.array (4,2), array that contain the distribution parameters of each configuration in the following order :
                    - theta_E : Einstein radius(angle), log-normal distribution [mu,sigma]- default : N(0,0.1)
                    - gamma   : Logarithmic slope of the power-law profile, log-normal distribution [mu,sigma]- default : N(0.7,0.1)
                    - q       : Axis ratio, uniform distribution [a,b] - default : U(0.7,1)
                    - phi     : Position angle, uniform distribution [a,b] - default : U(0,pi/2)
        :param mass_range : np.array (7,2), array that contain the distribution parameters of each source configurationin the following order :
                - amp      : Surface brightness/amplitude value at the half light radius, uniform distribution [a,b] - default : U(20,24)
                - R        : Semi-major axis half light radius, log-normal distribution [mu,sigma]- default : N(-0.7,0.4)
                - n        : Sersic index, log-normal distribution [mu,sigma]- default : N(0.7,0.4)
                - center_x : x-position of the source center, uniform distribution [a,b] - default : U(-0.5,0.5)
                - center_y : y-position of the source center, uniform distribution [a,b] - default : U(-0.5,0.5)
                - q        : Axis ratio, uniform distribution [a,b] - default : U(0.7,1)
                - phi      : Position angle, uniform distribution [a,b] - default : U(0,pi/2)
        """

        #define the psf and the type of observation (here HST)
        psf_hdu = fits.open('data/psf.fits')
        kwargs_hst = HST().kwargs_single_band()
        kwargs_hst['kernel_point_source'] = np.array(psf_hdu[0].data)
        self.numPix = numPix 

        kwargs_model = {'lens_model_list': ['PEMD'],                    #list of lens models to be used
                        'source_light_model_list': ['SERSIC_ELLIPSE'],  #list of extended source models to be used
                       }
        
        self.size = size
        self.image_config = SimAPI(numpix=numPix, kwargs_single_band=kwargs_hst, kwargs_model=kwargs_model)
        self.img_sim = self.image_config.image_model_class()
        
        
        #build dictionnary for the lens mass and the source model
        self.mass_error = LensMassError(size,seed, percent, center_x, center_y, mass_range)
        self.source_error = SourceError(size,seed, percent, source_range)
        
        self.metadata = pd.concat([self.mass_error.metadata,self.source_error.metadata], axis =1)
        self.metadata_error = pd.concat([self.mass_error.metadata_error,self.source_error.metadata_error], axis =1)
        self.images = np.zeros((self.size, 1,self.numPix, self.numPix))
        
        for i in np.arange(0,self.size): 
            kwargs_lens = self.mass_error.get_kwargs(i)
            kwargs_source = self.source_error.get_kwargs(i)

            #generate image
            self.images[i,:,:] = self.img_sim.image(kwargs_lens, kwargs_source, kwargs_ps=None)
        
    
class  Residual:
    """
    Class that build the residual maps and the metadata. This dataset ends up in the folder data/dataset by default.
    The data set is generated by the lenstronomy simulators. Error combinations are generated through a dictionnary.
    
    The file ID is coded as following : S[size]R[ratio]_lens.h5 for the residual maps and S[size]R[ratio]_meta.h5 
    for the metadata.
    """

    def build(self, size:int, ratio:float = 0.75, per_error:np.array =np.array([0.005, 0.015, 0.005]), num_label:int = 2, 
              center_x:float = 0, center_y:float = 0, mass_range = None, source_range = None, lower_chi:float = 1.2, upp_bound:float = 6, path_data:str = "data/dataset/"):
        """
        
        :param size         : int, size of the final data set
        :param ratio        : float, ratio between true and false residuals maps - default : ratio = 0.75
        :param per_error    : np.array(1,3), percentage of error between observed data and the model - default : per_error = [0.005, 0.015, 0.005]
        :param num_label    : int, number of labels - default : num_label = 2
        :param center_x     : float, x-position of the lens center - default : center_x = 0
        :param center_y     : float, y-position of the lens center - default : center_y = 0
        :param mass_range   : np.array (4,2), array that contain the distribution parameters of each configuration in the following order :
                    - theta_E : Einstein radius(angle), log-normal distribution [mu,sigma]- default : N(0,0.1)
                    - gamma   : Logarithmic slope of the power-law profile, log-normal distribution [mu,sigma]- default : N(0.7,0.1)
                    - q       : Axis ratio, uniform distribution [a,b] - default : U(0.7,1)
                    - phi     : Position angle, uniform distribution [a,b] - default : U(0,pi/2)
        :param source_range : np.array (7,2), array that contain the distribution parameters of each source configurationin the following order :
                - amp      : Surface brightness/amplitude value at the half light radius, uniform distribution [a,b] - default : U(20,24)
                - R        : Semi-major axis half light radius, log-normal distribution [mu,sigma]- default : N(-0.7,0.4)
                - n        : Sersic index, log-normal distribution [mu,sigma]- default : N(0.7,0.4)
                - center_x : x-position of the source center, uniform distribution [a,b] - default : U(-0.5,0.5)
                - center_y : y-position of the source center, uniform distribution [a,b] - default : U(-0.5,0.5)
                - q        : Axis ratio, uniform distribution [a,b] - default : U(0.7,1)
                - phi      : Position angle, uniform distribution [a,b] - default : U(0,pi/2)
        :param upp_bound    : float, upper amplitude of residuals map, to remove errors that are too obvious - default : upp_bound = 6
        :param path_data    : str, path where the binary file is saved - default : path_data = "data/dataset/"
        """

        metadata = pd.DataFrame()
        metadata_error = pd.DataFrame()
        percent_i = []
        rng = np.random.default_rng(2021)
        self.per_error = per_error
        bool_mdimg = []
        index_k = 0
        self.path_data = path_data
        self.size = round(size/(2**num_label))
        
        dataset_model = LensDataset(size = 1, percent = per_error[0], center_x = center_x, 
                                    center_y = center_y, mass_range = mass_range, source_range= source_range)
        self.channels = dataset_model.images.shape[1]
        self.input_size = dataset_model.numPix
        residuals = np.zeros((size,self.channels,self.input_size,self.input_size), dtype=np.half)
        
        #build residuals for each combination of error
        for errorID in range(2**num_label):
            
            index_i = 0; iter_i = 0;
            if errorID == 0:
                percent = 0; lower = -1; 
            else:
                percent = per_error[errorID-1]
                lower = lower_chi
                
            dataset_model = LensDataset(size = self.size, percent = percent, seed =  rng.integers(10000), center_x = center_x, 
                                        center_y = center_y, mass_range = mass_range, source_range= source_range)
            size_idx = self.size
            while iter_i < self.size:
                new_img, error = self._build_image_(dataset_model, index_i, errorID)
                if self._test_chi_ (new_img[0,:,:], lower, upp_bound):
                    iter_i +=1; 
                    residuals[index_k,:,:,:] = new_img
                    bool_mdimg.append(error)
                    metadata = pd.concat([metadata,dataset_model.metadata.take([index_i])])
                    metadata_error = pd.concat([metadata_error,dataset_model.metadata_error.take([index_i])])
                    percent_i.append(percent) 
                    index_k += 1;
                
                index_i += 1;
                if index_i >= size_idx : 
                    index_i = 0
                    size_idx = 500
                    dataset_model = LensDataset(size = size_idx, percent = percent, seed =  rng.integers(10000), center_x = center_x,
                                                center_y = center_y, mass_range = mass_range, source_range= source_range)
        
        #add the labels for each model and store the dataset in .h5 file    
        #use a multi label encoder - error is defined as a set of string
        mlb = MultiLabelBinarizer()
        bool_mdimg = mlb.fit_transform(bool_mdimg)
        metadata['class'] = np.reshape(bool_mdimg, (-1, num_label)).tolist()
        metadata_error['class'] =  np.reshape(bool_mdimg, (-1, num_label)).tolist()
        metadata['percent'] = percent_i
        metadata = metadata.reset_index()
        metadata_error = metadata.reset_index()
        self.residuals = residuals; self.metadata = metadata; self.metadata_error = metadata_error;
        
        #save the dataset
        ID_img = "S"+str(size)+"R"+str(int(ratio*100))
        store_hdf5(residuals, metadata, ID_img, path = self.path_data)
    
    def _build_image_(self, dataset_model:LensDataset, iter_i:int, errorID:int): 
        """
        
        :param dataset_model : LensDataset, object that contain all image configuration and images
        :param iter_i        : int, index of the used image
        :param errorID       : int, error ID to select which error dictionnary must be selected
        :return              : np.array(1,64,64), residual map in unit of noise
        """
        
        img_test = np.zeros((self.channels,dataset_model.numPix,dataset_model.numPix))  
        dict_err = self.dict_error(dataset_model, iter_i)
        kwargs_lens, kwargs_source, error =  dict_err[str(errorID)]
        img_test = dataset_model.img_sim.image(kwargs_lens, kwargs_source, kwargs_ps=None)

        image_model = dataset_model.images[iter_i]
        image_real = img_test + dataset_model.image_config.noise_for_model(model = img_test)
        sigma = np.sqrt(dataset_model.img_sim.Data.C_D_model(model = img_test))

        return (image_real-image_model)/sigma, error
        
        
    def dict_error(self, dataset_model:LensDataset, i:int):
        """
        
        Note : To add additionnal error the error list and the kwargs error must be redefined.
        
        :param dataset_model : lensdataset, object that contain all image configuration and images
        :param i             : int, index of the used image
        :return              : dictionnary, with every type of error combinations. One combination is a tuple with the parameters and an error set.
        """
        
        #list of possible errors
        error_list = [dataset_model.mass_error.type_error, dataset_model.source_error.type_error]
        kwargs_err = [dataset_model.mass_error.add_error(i), dataset_model.source_error.add_error(i)]
        
        #build all possible combinations of error 
        err = []
        for L in range(len(error_list)+1):
            err = err + list(combinations(error_list, L))
            
        err_set = [set(t) for t in err]
        kwargs = []
        dict_err = {}
        
        #build a dictionnary for the image i with every type of error
        for k, err_i in enumerate(err_set):
            kwargs_k =[dataset_model.mass_error.get_kwargs(i), dataset_model.source_error.get_kwargs(i)]
            
            for j, error_list_j in enumerate(error_list):
                if error_list_j in err_i:
                    kwargs_k[j] = kwargs_err[j]
            
            kwargs_k.append(err_i)
            dict_err[str(k)] = tuple(kwargs_k)

        return dict_err
    
    def _test_chi_ (self, images:np.array, lower_chi:float, upper_amp:float):
        """
        
        :param images     : np.array(1,64,64), image that is tested
        :param lower_chi  : float, greater than one, correspond to the lower bound of accepted noise
        :param upp_bound  : float, upper bound of amplitude to remove errors that are too obvious
        """
        chi = np.sum((images**2)/(images.shape[0]*images.shape[1]))
        bool_chi = False
        # check if the error is larger then maps
        if lower_chi < chi:
            # check if the error is not too obvious
            if np.count_nonzero(np.absolute(images[:,:]) > upper_amp)==0:
                bool_chi = True
            
        return bool_chi
            
        
        
class CombineDataset(Dataset):
    """
    This class helps us to build a pytorch tensor by combining the images, images'
    metadata and labels.
    
    """

    def __init__(self, frame, id_col, label_name:str, image:str, normalize:bool = True, nb_channel:int = 1)->None:
        """
        
        :param frame      : dataframe, frame with the tabular data.
        :param id_col     : string, name of the column that connects image to tabular data
        :param label_name : string, name of the column with the label to be predicted
        :param path_imgs  : string, path to the folder where the images are.
        :param nb_channel : int, number of channels.
        """
        self.frame = frame
        self.id_col = id_col
        self.label_name = label_name
        self.nb_channel = nb_channel
        
        #normalize image along channels
        if normalize:
            image = (image -image.mean(axis=(0,2,3), keepdims = True))/(image.std(axis=(0,2,3), keepdims = True))
        self.image = image
        
    def __len__(self)->int:
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
       
        
        #get labels
        label = np.array(self.frame[self.label_name].iloc[idx])
        label = torch.from_numpy(label.astype(np.float32))

        return  image, feats, label
