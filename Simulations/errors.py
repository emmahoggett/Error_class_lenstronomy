import numpy as np
import pandas as pd
import math
import random

class MassError():
    def __init__(self, size, seed, center_x = 0, center_y = 0, mass_param = None):
        np.random.seed(seed)
        self.seed = seed
        if mass_param == None:
            thetaE_range = [0.9, 1.2]; gamma_range = [1.9, 2]; q_range = [0.7, 1]; phi_range = [0, math.pi];
            self.mass_range = np.stack((thetaE_range,gamma_range,q_range,phi_range))
        elif type(mass_range) is np.ndarray:
            if mass_param.shape == (4,2):
                self.mass_param = mass_param
            else:
                raise Exception('mass_range should be a (4,2) numpy array. The shape of mass_range was: {}'.format(mass_param.shape))
        else:
            raise Exception('mass_range should be a numpy array. The type of mass_range was: {}'.format(type(mass_param)))
        
        self.size = size
        self.data = np.zeros((self.size,self.mass_range.shape[0]))
        self.center_x = center_x; self.center_y = center_y
            
        for i in np.arange(0,2):
            self.data[:,i] = np.random.uniform(self.mass_range[i,0],self.mass_range[i,1],self.size)
            
        q =  np.random.uniform(self.mass_range[2,0],self.mass_range[2,1],self.size)
        phi = np.random.uniform(self.mass_range[3,0],self.mass_range[3,1],self.size)
        self.data[:,2] = (1-q[:])/(1+q[:])*np.cos(phi[:])
        self.data[:,3] = (1-q[:])/(1+q[:])*np.sin(phi[:])
        
        
        self.metadata = pd.DataFrame(data=self.data)
        self.metadata.columns =  ['MASS_PROFILE_theta_E', 'MASS_PROFILE_gamma', 'MASS_PROFILE_e1', 'MASS_PROFILE_e2']
        self.k_seed = 0
        self.type_error = np.array([0,1,0])
        
    def get_kwargs(self, index):
        kwargs_spemd = {'theta_E': self.metadata['MASS_PROFILE_theta_E'][index], 'gamma': self.metadata['MASS_PROFILE_gamma'][index], 
                        'center_x': self.center_x, 'center_y': self.center_y, 
                        'e1':self.metadata['MASS_PROFILE_e1'][index], 'e2': self.metadata['MASS_PROFILE_e2'][index]}  
        return [kwargs_spemd]
    
    def add_error(self, index, percent = 0.1):
        self.k_seed = self.k_seed + 1
        random.seed(self.k_seed)
        range_data = self.mass_range[:,1] - self.mass_range[:,0]
        range_ellip = (1- self.mass_range[3,0])/(1 + self.mass_range[3,0])- (1- self.mass_range[3,1])/(1 + self.mass_range[3,1])
        
        kwargs_spemd = {'theta_E': self.metadata['MASS_PROFILE_theta_E'][index] + random.uniform(-1, 1)*percent*range_data[0], 'gamma': self.metadata['MASS_PROFILE_gamma'][index]+ random.uniform(-1, 1)*percent*range_data[1], 
                        'center_x': self.center_x, 'center_y': self.center_y, 
                        'e1':self.metadata['MASS_PROFILE_e1'][index] + random.uniform(-1, 1)*percent*range_ellip, 'e2': self.metadata['MASS_PROFILE_e2'][index] + random.uniform(0, 1)*percent*range_ellip}  
        return [kwargs_spemd]
    
    
class SourceError():
    def __init__(self, size, seed, source_param = None):
        np.random.seed(seed)
        self.seed = seed
        if source_param == None:
            amp_range = [20,24]; q_range = [0.7, 1]; phi_range = [0, math.pi]; Rsersic_range = [0.45,0.6]; nsersic_range = [1.35,3.05]; center_range = [-0.5,0.5]
            self.source_param = np.stack((amp_range, Rsersic_range, nsersic_range,center_range,center_range,q_range,phi_range))

        elif type(source_param) is np.ndarray:
            if source_param.shape == (7,2):
                self.source_param = source_param
            else:
                raise Exception('source_param should be a (7,2) numpy array. The shape of source_param was: {}'.format(source_param.shape))
        else:
            raise Exception('source_param should be a numpy array. The type of source_param was: {}'.format(type(source_param)))

        self.size = size
        self.data = np.zeros((self.size,self.source_param.shape[0]))

        for i in np.arange(0,5):
            self.data[:,i] = np.random.uniform(self.source_param[i,0],self.source_param[i,1],self.size)

        q =  np.random.uniform(self.source_param[5,0],self.source_param[5,1],self.size)
        phi = np.random.uniform(self.source_param[6,0],self.source_param[6,1],self.size)
        self.data[:,5] = (1-q[:])/(1+q[:])*np.cos(phi[:])
        self.data[:,6] = (1-q[:])/(1+q[:])*np.sin(phi[:])


        self.metadata = pd.DataFrame(data=self.data)
        self.metadata.columns =  ['SOURCE_PROFILE_amp', 'SOURCE_PROFILE_R_sersic', 'SOURCE_PROFILE_n_sersic','SOURCE_PROFILE_center_x', 
                                'SOURCE_PROFILE_center_y', 'SOURCE_PROFILE_e1', 'SOURCE_PROFILE_e2']
        self.k_seed = 0
        self.type_error = np.array([0,0,1])

    def get_kwargs(self, index):
        # Sersic parameters in the initial simulation for the source
        kwargs_sersic = {'amp':  self.metadata['SOURCE_PROFILE_amp'][index], 'R_sersic': self.metadata['SOURCE_PROFILE_R_sersic'][index],
                         'n_sersic':  self.metadata['SOURCE_PROFILE_n_sersic'][index], 'e1':  self.metadata['SOURCE_PROFILE_e1'][index], 
                         'e2':  self.metadata['SOURCE_PROFILE_e2'][index], 'center_x': self.metadata['SOURCE_PROFILE_center_x'][index], 
                         'center_y':  self.metadata['SOURCE_PROFILE_center_y'][index]}
        return [kwargs_sersic]
    
    def add_error(self, index, percent = 0.1):
        self.k_seed = self.k_seed + 1
        random.seed(self.k_seed)
        range_data = self.source_param[:,1] - self.source_param[:,0]
        range_ellip = (1- self.source_param[5,0])/(1 + self.source_param[5,0])- (1- self.source_param[5,1])/(1 + self.source_param[5,1])
        kwargs_sersic = {'amp':  self.metadata['SOURCE_PROFILE_amp'][index]+ np.random.uniform(-1, 1)*percent*range_data[0], 'R_sersic': self.metadata['SOURCE_PROFILE_R_sersic'][index] + np.random.uniform(-1, 1)*percent*range_data[1],
                         'n_sersic':  self.metadata['SOURCE_PROFILE_n_sersic'][index]+ np.random.uniform(-1, 1)*percent*range_data[2], 'e1':  self.metadata['SOURCE_PROFILE_e1'][index]+ np.random.uniform(-1, 1)*percent*range_ellip, 
                         'e2':  self.metadata['SOURCE_PROFILE_e2'][index]+ np.random.uniform(0, 1)*percent*range_ellip , 'center_x': self.metadata['SOURCE_PROFILE_center_x'][index]+ np.random.uniform(-1, 1)*percent*range_data[3], 
                         'center_y':  self.metadata['SOURCE_PROFILE_center_y'][index] + np.random.uniform(-1, 1)*percent*range_data[4]}
        return [kwargs_sersic]