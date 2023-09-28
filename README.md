# Thesis-NN-Bridges

Thesis project about bridges crack detection using image recognition

## Setup

|        | Version |
|--------|---------|
| Python | 3.10    |

## Environment

In the Environment folder there is an environment.yml to create an environment with all the libraries required for the
project to work.<br/>
Updates will be defined in the README file in the Environment folder.
A requirements.txt is also available for ad hoc installations.

## Dataset

The used dataset for this project can be located in the Dataset folder and might receive updates during the development
of this project (dataset is around 250MB, and can be downloaded from the link below).<br/>
If you follow the Notebooks we higly recommend to save the dataset in the 'Dataset/Train' folder.<br/>
There are 2 free-to-use test images in the 'Dataset/Test' folder you can use to test the model.<br/>
<br/>
The starting dataset is: [Default Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## Notebooks

There are two notebooks that shows how to train and how to use a model for inference.

# Code

The main code is under the 'src' folder.<br/>
There are 2 scripts called 'train.py' and 'infer.py' that can be used to train or perform an inference.<br/>
In the 'utils' folder there are all the functions used for the training, the model management etc...

## Results

The weights provided give a 97.41% accuracy on the validation set (2 classes: Normal vs Crack).<br/>
We trained the model for 3 epochs (su further upgrades are easily feasible).

### Python Environment

* Install conda/anaconda distribution

* Create a new conda environment:

  > `Modify the last row of environment.yml so it links the environment into your 'envs' pc account directory`
  >
  > `conda env create -f Environment/environment.yml`
  >
  > `conda activate bridges_crack_detection`

* Test if all dependencies are satisfied (make you sure to be in the repository directory, all commands in this guide 
* are relative to the repository):
  > `cd Environment`

  > `python test_dependencies.py`

  If *"All dependencies are satisfied"* message shows up, everything has been setup correctly!.

  Remember to **activate** bridges_crack_detection conda environment before writing any code!<br/>
* To deactivate:
  > `conda deactivate`

* To delete the environment (only the environment is deleted, the project directory is a separate entity!):
  > `conda deactivate`
  >
  > `conda env remove --name bridges_crack_detection`
