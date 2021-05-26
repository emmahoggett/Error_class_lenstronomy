# LASTRO Classify errors from strong lens modelling with neural networks

### Description
Strong gravitational lensing modelling is a powerful method to scrutinize an object's unknown astrophysical and cosmological properties and enable a quantization of dark matter in the universe. Nevertheless, insidious error often riddles generated models. This project investigates an efficient error classifier on a benchmarked data set. Our analysis shines a light on blended combinations of neural networks. 

### Getting Started
This version was designed for python 3.6.6 or higher. The code is stored in the folder `Simulations/`. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is python `run.py`.

### Prerequisites


#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [scikit-learn](https://scikit-learn.org/): `pip install -U scikit-learn`
* [pytorch](https://pytorch.org/): `pip3 install torch torchvision torchaudio`
* [skmultilearn](http://scikit.ml/): `pip install scikit-multilearn`
* [h5py](https://docs.h5py.org/en/latest/build.html): `pip install h5py`

The `fastell` library allows us to use the `PEMD` mass model.
Strong gravitationnal lens libraries:
* [lenstronomy](https://pypi.org/project/lenstronomy/): `pip lenstronomy`
* [fastell](https://github.com/sibirrer/fastell4py): `python setup.py install --user`



#### Code
To launch the code `run.py` use the following codes:
* `helpers/data_generation/error_generation_chi2.py`: helpers to generate the residual maps with MSE and maximal amplitude consideration.
* `helpers/data_generation/errors.py`: helpers that contains class definitions of error.
* `helpers/data_generation/file_management.py`: helpers to read and create `.hdf5` files.
* `helpers/model/`: contain the classes for define the neural networks.
* `helpers/model/helpers_model.py`: handle neural network models training, testing, saving and loading.

#### Files
As the Hubble Space Telescope(HST) model is used, it is necessary to provide a point-spread function(PSF). To do so, a PSF file from the [Time Delay Lens Modeling Challenge](https://tdlmc.github.io/) is selected, which is `rung0/code1/f160w-seed3/drizzled_image/psf.fits`. 

Simulated neural networks model :
* `data/model/checkpoints/AlexNet_optimal.pt` : optimal trained AlexNet
* `data/model/checkpoints/ResNet18_optimal.pt` : optimal trained ResNet18
* `data/model/checkpoints/VGG11_optimal.pt` : optimal trained VGG11
* `data/model/checkpoints/SqueezeNet_optimal.pt` : optimal trained SqueezeNet
* `data/model/checkpoints/GoogleNet_optimal.pt` : optimal trained VGG11
* `data/model/checkpoints/DenseNet161_optimal.pt` : optimal trained SqueezeNet

In this folder (`data/model/checkpoints/`), models are also saved under the name `_current` which correspond to the neural network trained over 50 epochs.


#### Deeplenstronomy wrapper
`deeplenstronomy` is a wrapper of `lenstronomy` that enable a large generation of simulated strong gravitational lens. To use it, a yaml file must be filled by the user with the desired strong lens configurations. `lenstronomy` and `fastell` must be installed. With this wrapper, the point-spread function(PSF) for an HST model 

To use the deeplenstronomy wrapper, the following library are installed :
*[deeplenstronomy](https://pypi.org/project/deeplenstronomy/): `pip install deeplenstronomy`
*[PyYAML](https://pypi.org/project/PyYAML/): `pip install PyYAML`

`deeplenstronomy` is used in the folder `Simulations/deeplens/` and contain helpers to generate residual maps and a notebook `deeplens.ipynb` that contain some basic observations over the generated data.

##### YAML examples
The yaml files are build according to the [DeepLenstronomy Configuration Files](https://deepskies.github.io/deeplenstronomy/Notebooks/ConfigFiles.html) instructions. Three examples are available in the section `deeplens/data/configFile/`.

### Additional content
The `datanalysis.ipynb` and `datanalysis_chi2.py` are a notebook that contains some basic observations over the generated residuals and images. The first one have no consideration of noise-like and error-obvious maps, while the second take those considerations into account.
The folder `Papers` contains scientific papers that inspired our project.

### Documentation


### Author
* Hoggett Emma <emma.hoggett@epfl.ch>


### Project Status
The project was submitted on the 11 June 2021, as part of a semester project in the [LASTRO](https://www.epfl.ch/labs/lastro/) laboratory at EPFL. This project was supervised by:
 * Pr. Courbin Frédéric <frederic.courbin@epfl.ch>
 * Dr. Peel Austin <austin.peel@epfl.ch>
 * PhD. Galan Aymeric <aymeric.galan@epfl.ch>
