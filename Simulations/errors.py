#  this file contains classes to build lens mass and source light dictionnaries

import numpy as np
import pandas as pd
import math
import random


class LensMassError():
    """ 
    Class that generate the lens mass dictionnary and handle lens mass error residuals dictionnary.
    
    The Einstein ring parameter converts to the definition used by GRAVLENS as follow:
    (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]
    
    The fastell wrapper must be installed to use this class as it uses the PEMD mass model. 
    The default model configuration is the following :
        - Mass model               : PEMD
        - Einstein radius, theta_E : log-normal distribution N(0,0.1)
        - Logarithmic slope, gamma : log-normal distribution N(0.7,0.1)
        - Axis ratio, q            : uniform distribution U(0.7,1)
        - Position angle, phi      : uniform distribution U(0,pi/2)
        - X-position, center_x     : 0
        - Y-position, center_y     : 0
        
    """
    def __init__(self, size:int, seed:int, center_x:float = 0, center_y:float = 0, mass_param = None):
        """
        
        :param size: int, size of the data generated
        :param seed: int, random seed
        :param center_x: float, x-position of the lens center 
        :param center_y: float, y-position of the lens center
        :param mass_param: np.array (4,2), array that contain the distribution parameters of each configuration in the following order :
                    - theta_E : Einstein radius(angle), log-normal distribution [mu,sigma]- default : N(0,0.1)
                    - gamma   : Logarithmic slope of the power-law profile, log-normal distribution [mu,sigma]- default : N(0.7,0.1)
                    - q       : Axis ratio, uniform distribution [a,b] - default : U(0.7,1)
                    - phi     : Position angle, uniform distribution [a,b] - default : U(0,pi/2)
        """
        np.random.seed(seed)
        self.seed = seed
        
        if mass_param == None:
            thetaE_range = [0, 0.1]; gamma_range = [0.7, 0.1]; q_range = [0.7, 1]; phi_range = [0, math.pi/2];
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
            self.data[:,i] = np.random.lognormal(self.mass_range[i,0],self.mass_range[i,1],self.size)
            
        q =  np.random.uniform(self.mass_range[2,0],self.mass_range[2,1],self.size)
        phi = np.random.uniform(self.mass_range[3,0],self.mass_range[3,1],self.size)
        self.data[:,2] = (1-q[:])/(1+q[:])*np.cos(2*phi[:])
        self.data[:,3] = (1-q[:])/(1+q[:])*np.sin(2*phi[:])
        
        
        self.metadata = pd.DataFrame(data=self.data)
        self.metadata.columns =  ['MASS_PROFILE_theta_E', 'MASS_PROFILE_gamma', 'MASS_PROFILE_e1', 'MASS_PROFILE_e2']
        self.k_seed = 0
        self.type_error = 'lensmass'
        
    def get_kwargs(self, index:int):
        """
        
        :param index : int, index of the sample
        :return      : dictionnary, simulated kwargs with no error for the lens mass model
        """
        kwargs_spemd = {'theta_E': self.metadata['MASS_PROFILE_theta_E'][index], 'gamma': self.metadata['MASS_PROFILE_gamma'][index], 
                        'center_x': self.center_x, 'center_y': self.center_y, 
                        'e1':self.metadata['MASS_PROFILE_e1'][index], 'e2': self.metadata['MASS_PROFILE_e2'][index]}  
        return [kwargs_spemd]
    
    def add_error(self, index:int, percent:float = 0.1):
        """
        
        :param index   : int, index of the sample
        :param percent : float, percentage of relative error
        :return        : dictionnary, error mass dictionnary defined by the current index value with a relative error. The relative error is an uniform distribution U(-percent,percent).
        """
        self.k_seed = self.k_seed + 1
        random.seed(self.k_seed)
        
        kwargs_spemd = {'theta_E': self.metadata['MASS_PROFILE_theta_E'][index] + random.uniform(-1, 1)*percent*abs(self.metadata['MASS_PROFILE_theta_E'][index]), 
                        'gamma': self.metadata['MASS_PROFILE_gamma'][index]+ random.uniform(-1, 1)*percent*abs(self.metadata['MASS_PROFILE_gamma'][index]), 
                        'center_x': self.center_x, 'center_y': self.center_y, 
                        'e1':self.metadata['MASS_PROFILE_e1'][index] + random.uniform(-1, 1)*percent*abs(self.metadata['MASS_PROFILE_e1'][index]), 
                        'e2': self.metadata['MASS_PROFILE_e2'][index] + random.uniform(-1, 1)*percent*abs(self.metadata['MASS_PROFILE_e2'][index])}  
        return [kwargs_spemd]
    
    
class SourceError():
    """
    Class that generate the source dictionnary and handle source error residuals dictionnary.

    .. math::

        I(R) = I_0 \exp \left[ -b_n (R/R_{\rm Sersic})^{\frac{1}{n}}\right]

    with :math:`I_0 = amp`
    and
    with :math:`b_{n}\approx 1.999\,n-0.327`
    
    
    The default model configuration is the following :
        - Source model                      : Sersic
        - Surface brightness/amplitude, amp : uniform distribution U(20,24)
        - Semi-major axis half light, R     : log-normal distribution N(-0.7,0.4)
        - Sersic index, n                   : log-normal distribution N(0.7,0.4)
        - X-position, center_x              : uniform distribution U(-0.5,0.5)
        - Y-position, center_y              : uniform distribution U(-0.5,0.5)
        - Axis ratio, q                     : uniform distribution U(0.7,1)
        - Position angle, phi               : uniform distribution U(0,pi/2)
        """
    def __init__(self, size:int, seed:int, source_param = None):
        """  
        
        :param size: int, size of the data generated
        :param seed: int, random seed
        :param source_param: np.array (7,2), array that contain the distribution parameters of each source configurationin the following order :
                - amp      : Surface brightness/amplitude value at the half light radius, uniform distribution [a,b] - default : U(20,24)
                - R        : Semi-major axis half light radius, log-normal distribution [mu,sigma]- default : N(-0.7,0.4)
                - n        : Sersic index, log-normal distribution [mu,sigma]- default : N(0.7,0.4)
                - center_x : x-position of the source center, uniform distribution [a,b] - default : U(-0.5,0.5)
                - center_y : y-position of the source center, uniform distribution [a,b] - default : U(-0.5,0.5)
                - q        : Axis ratio, uniform distribution [a,b] - default : U(0.7,1)
                - phi      : Position angle, uniform distribution [a,b] - default : U(0,pi/2)
        """
        np.random.seed(seed)
        self.seed = seed
        
        if source_param == None:
            amp_range = [20,24]; q_range = [0.7, 1]; phi_range = [0, math.pi]; Rsersic_range = [-0.7, 0.4]; nsersic_range = [0.7,0.4]; center_range = [-0.5,0.5]
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
            
        self.data[:,1] = np.random.lognormal(self.source_param[1,0],self.source_param[1,1],self.size)
        self.data[:,2] = np.random.lognormal(self.source_param[2,0],self.source_param[2,1],self.size)

        q =  np.random.uniform(self.source_param[5,0],self.source_param[5,1],self.size)
        phi = np.random.uniform(self.source_param[6,0],self.source_param[6,1],self.size)
        self.data[:,5] = (1-q[:])/(1+q[:])*np.cos(phi[:])
        self.data[:,6] = (1-q[:])/(1+q[:])*np.sin(phi[:])


        self.metadata = pd.DataFrame(data=self.data)
        self.metadata.columns =  ['SOURCE_PROFILE_amp', 'SOURCE_PROFILE_R_sersic', 'SOURCE_PROFILE_n_sersic','SOURCE_PROFILE_center_x', 
                                'SOURCE_PROFILE_center_y', 'SOURCE_PROFILE_e1', 'SOURCE_PROFILE_e2']
        self.k_seed = 0
        self.type_error = 'source'

    def get_kwargs(self, index:int):
        """
        
        :param index : int, index of the sample
        :return      : dictionnary, simulated kwargs with no error for the source light model
        """
        kwargs_sersic = {'amp':  self.metadata['SOURCE_PROFILE_amp'][index], 'R_sersic': self.metadata['SOURCE_PROFILE_R_sersic'][index],
                         'n_sersic':  self.metadata['SOURCE_PROFILE_n_sersic'][index], 'e1':  self.metadata['SOURCE_PROFILE_e1'][index], 
                         'e2':  self.metadata['SOURCE_PROFILE_e2'][index], 'center_x': self.metadata['SOURCE_PROFILE_center_x'][index], 
                         'center_y':  self.metadata['SOURCE_PROFILE_center_y'][index]}
        return [kwargs_sersic]
    
    def add_error(self, index:int, percent:float = 0.1):
        """
        
        :param index   : int, Index of the sample
        :param percent : float, Percentage of relative error
        :return        : dictionnary, Error source dictionnary defined by the current index value with a relative error. The relative error is an uniform distribution U(-percent,percent).
        """
        self.k_seed = self.k_seed + 1
        random.seed(self.k_seed)
        
        kwargs_sersic = {'amp':  self.metadata['SOURCE_PROFILE_amp'][index]+ np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_amp'][index]), 
                         'R_sersic': self.metadata['SOURCE_PROFILE_R_sersic'][index] + np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_R_sersic'][index]),
                         'n_sersic':  self.metadata['SOURCE_PROFILE_n_sersic'][index]+ np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_n_sersic'][index]),
                         'e1':  self.metadata['SOURCE_PROFILE_e1'][index]+ np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_e1'][index]), 
                         'e2':  self.metadata['SOURCE_PROFILE_e2'][index]+ np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_e2'][index]) , 
                         'center_x': self.metadata['SOURCE_PROFILE_center_x'][index]+ np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_center_x'][index]), 
                         'center_y':  self.metadata['SOURCE_PROFILE_center_y'][index] + np.random.uniform(-1, 1)*percent*abs(self.metadata['SOURCE_PROFILE_center_y'][index])}
        return [kwargs_sersic]