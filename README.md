# EPFL Machine Learning Recommender System 2019

### Description
This project's aim is to create predict good recommandation for films. Each user gives appreciations on films with grades that are integers between 0 and 5. One has to predict the remaining grades.



### Getting Started
This version was designed for python 3.6.6 or higher. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is `python run.py`. The code should return a `results.csv` file with all its predictions, from the test data.

### Prerequisites

#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [pytorch](https://surprise.readthedocs.io/en/stable/index.html): `pip install scikit-surprise`
To install 
* [lenstronomy](https://scikit-learn.org/stable/): `pip lenstronomy`
* [deeplenstronomy](https://keras.io/): `pip install Keras`


#### Code
To launch the code `run.py` use the following codes and pickle files:
* `helpers.py` : Deal with creation and loading of `.csv` files



The `data` folder is also needed to store training data, the data for the final submission and the test set trained on 0.8 of the training set, which will be used for the ridge regression : `data_train.csv`, `sampleSubmission.csv` and `test_pred.pickle`.

### Additional content
The folder `models` contains python code that established our machine learning procedure,  contains the testing of the different methods implemented. Those files are run into the main code, which is `run.py`

The folder `papers` contains scientific papers that inspired our project.

### Documentation


### Authors
* Hoggett Emma <emma.hoggett@epfl.ch>

### Project Status
The project was submitted on the 19 June 2021, as part of 
