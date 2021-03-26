import numpy as np
import pandas as pd
import random
import deeplenstronomy.deeplenstronomy as dl
import lenstronomy.Util.image_util as image_util
from helpers import*



col_drop = ['OBJID-g', 'H0-g', 'Om0-g', 'exposure_time-g', 'numPix-g',
            'pixel_scale-g', 'psf_type-g', 'read_noise-g', 'ccd_gain-g', 
            'seeing-g', 'magnitude_zero_point-g', 'sky_brightness-g', 
            'num_exposures-g', 'NUMBER_OF_NOISE_SOURCES-g', 'CONFIGURATION_LABEL-g',
            'CONFIGURATION_NAME-g', 'NUMBER_OF_PLANES-g','PLANE_1-NUMBER_OF_OBJECTS-g', 
            'PLANE_1-OBJECT_1-NAME-g', 'PLANE_2-NUMBER_OF_OBJECTS-g', 
            'PLANE_2-OBJECT_1-NAME-g', 'PLANE_1-OBJECT_1-REDSHIFT-g',
            'PLANE_1-OBJECT_1-NUMBER_OF_LIGHT_PROFILES-g','PLANE_1-OBJECT_1-NUMBER_OF_SHEAR_PROFILES-g',
            'PLANE_1-OBJECT_1-NUMBER_OF_MASS_PROFILES-g', 'PLANE_1-OBJECT_1-HOST-g',
            'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-NAME-g', 'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-magnitude-g',
            'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-center_x-g', 'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-center_y-g', 
            'PLANE_2-OBJECT_1-REDSHIFT-g', 'PLANE_2-OBJECT_1-NUMBER_OF_LIGHT_PROFILES-g', 
            'PLANE_2-OBJECT_1-NUMBER_OF_SHEAR_PROFILES-g', 'PLANE_2-OBJECT_1-NUMBER_OF_MASS_PROFILES-g', 
            'PLANE_2-OBJECT_1-HOST-g', 'PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g', 
            'PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g',
            'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-NAME-g', 'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g',
            'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_x-g', 'PLANE_2-OBJECT_1-LIGHT_PROFILE_1-center_y-g', 
            'BACKGROUND_IDX-g', 'PLANE_1-REDSHIFT-g', 'PLANE_2-REDSHIFT-g', 'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g',
            'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g', 'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-e1-g',
            'PLANE_1-OBJECT_1-LIGHT_PROFILE_1-e2-g','PLANE_1-OBJECT_1-MASS_PROFILE_1-NAME-g']

    
    
class  ResidualDeepLens:

    def __init__(self, path_config_model):
        """
        Args:
            path_config_model (string): 
        """
        random.seed(2)
        self.path_config_model = path_config_model

    def build(self, errorID, path_data=hdf5_dir, exp_time = 5400., background_rms = .005):
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

        self.size = dataset_model.CONFIGURATION_1_images.shape[0]
        self.channels = dataset_model.CONFIGURATION_1_images.shape[1]

        if errorID == 1:
            error = np.array([0,1,0])
        elif errorID == 2:
            error = np.array([0,0,1])
        else:
            error = np.array([0,1,1])
        
        metadata = pd.DataFrame()
        residuals = np.zeros((self.size*4, 1,64,64))
        k = 0
        
        for i in np.arange(0,self.size):
            metadata_temp = pd.concat([dataset_model.CONFIGURATION_1_metadata.take([i]).drop(col_drop,axis=1)]*(4), 
                                                ignore_index=True)
            bool_mdimg = np.array([], dtype='int')
            
            test = np.array([i])
            while test.shape[0]!=4:
                r=random.randint(1,self.size-1)
                if r not in test: test = np.append(test, r)
                
            for j in test:
                if i!=j:
                    bool_mdimg = np.concatenate((bool_mdimg,error))
                else:
                    bool_mdimg = np.concatenate((bool_mdimg,np.array([0,0,0])))
                # Residual between two images i and j
                for i_ch in np.arange(0,self.channels):
                    image_model = dataset_model.CONFIGURATION_1_images[i][i_ch]
                    poisson = image_util.add_poisson(image_model, exp_time=exp_time)
                    bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
                    image_real = image_model + poisson + bkg
                    residuals[k,i_ch,:,:] =  image_real-image_model
                k = k+1
                
            # Add the type of error in the metadata and the ID of the image
            metadata_temp['class'] =np.reshape(bool_mdimg, (-1, 3)).tolist()
            
            metadata = pd.concat([metadata,metadata_temp])
        store_hdf5(residuals, metadata, errorID)

