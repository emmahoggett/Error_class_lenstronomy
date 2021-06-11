# LASTRO Classifying errors from strong lens modelling with neural networks

### Description
Strong gravitational lensing modelling is a powerful method to scrutinize an object's unknown astrophysical and cosmological properties and enables a quantization of dark matter in the universe. Nevertheless, insidious errors often riddle generated models that need to be spot manually. Due to an increase in acquisition efficiency, this process becomes unpractical. Thereby, this project uses neural networks to classify residual maps as a key product of any lens modelling pipeline. The main application of such classifiers is to quickly and reliably obtain guidance for further refinement of a lens model while providing a study of degeneracies between model parameters. Our analysis shines a light on blended combinations of neural networks with an AUROC prediction score of 0.950. 


### Getting Started
This version was designed for python 3.6.6 or higher. The code is stored in the folder `Simulations/`. To run the model's calculation, it is only needed to execute the file `blending.ipynb` in Jupyter lab or Jupyter Notebook. To run the convolutional neural network models, launch the file `traincnn.ipynb` and ensure that the folder `model/checkpoints/` contain all used [models](https://drive.google.com/drive/folders/1TRWzx7hzKoBBLbV-Yk24aagejHkwG_aP?usp=sharing). If it is not the case, remove the following lines before each training :

```python
netbasic.load_checkpoint('_current') 
```
### Prerequisites


#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [scikit-learn](https://scikit-learn.org/): `pip install -U scikit-learn`
* [pytorch](https://pytorch.org/): `pip3 install torch torchvision torchaudio`
* [h5py](https://docs.h5py.org/en/latest/build.html): `pip install h5py`

The `fastell` library allows us to use the `PEMD` mass model.
Strong gravitationnal lens libraries:
* [lenstronomy](https://pypi.org/project/lenstronomy/): `pip install lenstronomy`
* [fastell](https://github.com/sibirrer/fastell4py): The repository must be cloned on the used computer
```python
 cd fastell4py
 python setup.py install --user
```



#### Code
To launch the code `blending.ipynb`, use the following codes:
* `helpers/data_generation/error_generation_chi2.py`: helpers to generate the residual maps with chi-squared and maximal amplitude consideration.
* `helpers/data_generation/errors.py`: helpers that contain class definitions of error.
* `helpers/data_generation/file_management.py`: helpers to read and create `.hdf5` files.
* `helpers/model/`: contain the classes for defining the neural networks.
* `helpers/model/helpers_model.py`: handle neural network models training, testing, saving and loading.

#### Files
As the Hubble Space Telescope(HST) model is used, it is necessary to provide a point-spread function(PSF). To do so, a PSF file from the [Time Delay Lens Modeling Challenge](https://tdlmc.github.io/) is selected, which is `rung0/code1/f160w-seed3/drizzled_image/psf.fits`. We selected this PSF as we deal with pixelized HST-like observation with the following configuration :

> Exposure time : 5400 [s]
>
> Sky brightness: 22
> 
> Number of exposure: 1
> 
> Seeing: None, the seeing is set by the PSF and correspond to the image resolution in arcsecond. For HST-like observation, the value should be around 0.08''
> 
> PSF type: PIXEL

[Simulated neural networks model](https://drive.google.com/drive/folders/1TRWzx7hzKoBBLbV-Yk24aagejHkwG_aP?usp=sharing) :
* `data/model/checkpoints/BasicCNN_optimal.pt` : optimal trained BasicCNN
* `data/model/checkpoints/AlexNet_optimal.pt` : optimal trained AlexNet
* `data/model/checkpoints/ResNet18_optimal.pt` : optimal trained ResNet18
* `data/model/checkpoints/VGG11_optimal.pt` : optimal trained VGG11
* `data/model/checkpoints/SqueezeNet_optimal.pt` : optimal trained SqueezeNet
* `data/model/checkpoints/GoogleNet_optimal.pt` : optimal trained VGG11
* `data/model/checkpoints/DenseNet161_optimal.pt` : optimal trained SqueezeNet

In this folder (`data/model/checkpoints/`), models are also saved under the name `_current`, corresponding to the neural network trained over 50 or 30 epochs.


#### Deeplenstronomy wrapper
`deeplenstronomy` is a wrapper of `lenstronomy` that enable a large generation of a simulated strong gravitational lens. To use it, a YAML file must be filled by the user with the desired strong lens configurations. `lenstronomy` and `fastell` must be installed. With this wrapper, the point-spread function(PSF) for an HST observation model is not required.

To use the deeplenstronomy wrapper, the following library are installed :
* [deeplenstronomy](https://pypi.org/project/deeplenstronomy/): `pip install deeplenstronomy`
* [PyYAML](https://pypi.org/project/PyYAML/): `pip install PyYAML`

`deeplenstronomy` is used in the folder `Simulations/deeplens/` and contain helpers to generate residual maps and a notebook `deeplens.ipynb` that contain some basic observations over the generated data.

##### YAML examples
The YAML files are built according to the [DeepLenstronomy Configuration Files](https://deepskies.github.io/deeplenstronomy/Notebooks/ConfigFiles.html) instructions. Three examples are available in the section `deeplens/data/configFile/`.

### Additional content
The `datanalysis.ipynb` and `datanalysis_chi2.ipynb` are a notebook that contains some basic observations over the generated residuals and images. The first one has no consideration of noise-like and error-obvious maps, while the second take those considerations into account. The notebook `optimizer.ipynb` contains the code that performed the analysis over optimizers, while `ratio_percent_baseline.py` study the effect on imbalance and percentage errors.

The dataset generated by lenstronomy wrapper deeplenstronomy is contained in the notebook `deeplens/deeplens.ipynb`. The notebook works with three `.yaml` files in the section `deeplens/data/configFile/`. The file names are the following 
* `config_model1.yaml`: contain a lens mass model with given distributions and a fixed source light model
* `config_model2.yaml`: contain a source light model with given distributions and a fixed lens mass model
* `config_model3.yaml`: contain a lens mass model and a source light model with given distributions

Finally, the folder `interpol/` contains a notebook to generate real source galaxies. The files `data/source_img/Images_Filtered_Rotated.npy` and `data/source_img/Labels_Filtered.csv` to run this notebook properly. The files are available [here](https://drive.google.com/drive/folders/13Iz3ZzmgKZI9KMCkAvcGPvNe08dD8183?usp=sharing).



### Documentation
Papers used for the project are contained in the folder `Papers/`. This folder also contains the report for this project.

### Author
* Hoggett Emma <emma.hoggett@epfl.ch>


### Project Status
The project was submitted on 11 June 2021, as part of a semester project in the [LASTRO](https://www.epfl.ch/labs/lastro/) laboratory EPFL. This project was supervised by:
 * Pr. Courbin Frédéric <frederic.courbin@epfl.ch>
 * Dr. Peel Austin <austin.peel@epfl.ch>
 * PhD. Galan Aymeric <aymeric.galan@epfl.ch>
