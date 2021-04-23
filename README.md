# LASTRO Classify errors from strong lens modelling with neural networks

### Description
Strong lens modelling is an usefull tool to 


### Getting Started
This version was designed for python 3.6.6 or higher. The code is stored in the folder `Simulations/`. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is python `run.py`.

### Prerequisites


#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [scikit-learn](https://scikit-learn.org/): `pip install -U scikit-learn`
* [pytorch](https://pytorch.org/): `pip3 install torch torchvision torchaudio`

The `fastell` library allows us to use the `PEMD` mass model.
Strong gravitationnal lens libraries:
* [lenstronomy](https://pypi.org/project/lenstronomy/): `pip lenstronomy`
* [fastell](https://github.com/sibirrer/fastell4py): `python setup.py install --user`



#### Code
To launch the code `run.py` use the following codes and pickle files:
* `helpers.py`: 
* `lenshelpers.py`: contain the 
* `errors.py`: contain the classes that build dictionnaries and error dictionnaries.

#### Files
As the Hubble Space Telescope(HST) model is used, it is necessary to provide a point-spread function(PSF). To do so, a PSF file from the [Time Delay Lens Modeling Challenge](https://tdlmc.github.io/) is selected, which is `rung0/code1/f160w-seed3/drizzled_image/psf.fits`. 


#### Deeplenstronomy wrapper
`deeplenstronomy` is a wrapper of `lenstronomy` that enable a large generation of simulated strong gravitational lens. To use it, a yaml file must be filled by the user with the desired strong lens configurations. `lenstronomy` and `fastell` must be installed. With this wrapper, the point-spread function(PSF) for an HST model 

To use the deeplenstronomy wrapper, the following library are installed :
*[deeplenstronomy](https://pypi.org/project/deeplenstronomy/): `pip install deeplenstronomy`

`deeplenstronomy` is used in the folder `Simulations/deeplens/` and contain helpers to generate residual maps and a notebook `deeplens.ipynb` that contain some basic observations over the generated data.

##### YAML examples
The yaml files are build according to the [DeepLenstronomy Configuration Files](https://deepskies.github.io/deeplenstronomy/Notebooks/ConfigFiles.html) instructions. Three examples 

### Additional content
The `datanalysis.ipynb` is a notebook that contains some basic observations over the generated residuals and images
The folder `Papers` contains scientific papers that inspired our project.

### Documentation


### Author
* Hoggett Emma <emma.hoggett@epfl.ch>


### Project Status
The project was submitted on the 19 June 2021, as part of a semester project in the LASTRO laboratory. This project was supervised by:
 * Pr. Courbin
 * Dr.
