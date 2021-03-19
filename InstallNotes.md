# Install Notes
These notes should aid installation of the pre-requisite Python packages into an Anaconda virtual environment on Windows. This should include all the packages needed, but check the installed packages against `requirements.txt` in case anything is missing.

## Pre-requisites
Assumes Anaconda has been installed. Visual basic is also needed for CUDA

## Create a python virtual environment
```
conda create --name LFMC python=3.8
conda activate LFMC
conda install -c anaconda cudatoolkit=10.1
conda install -c anaconda cudnn=7.6.5=cuda10.1_0
conda install pip
pip install tensorflow==2.3
```

## Check tensorflow works
In a python console run:
```
import tensorflow as tf
tf.__version__

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
```

## Install other packages
```
conda install pandas ipykernel jupyter matplotlib xlrd openpyxl scikit-learn scipy pydot
conda install -c conda-forge earthengine-api
earthengine authenticate
```
- This opens a browser window that requires you to sign in to google and then displays a verification code that needs to be copied and pasted into the command window at the prompt.

## Add the virtual environment to Jupyter, if not already there:
- `python -m ipykernel install --user --name=LFMC`

## Install JupyterLab and Spyder using Anaconda navigator
- Start menu shortcuts should be created automatically for Jupyter and Spyder, but change them to have the correct root directory - change the "%USERPROFILE%/" at the end of the Target to "<required directory>/"
- To create a shortcut for JupyterLab, copy and rename the Jupyter shortcut, and change "jupyter-notebook-script.py" to "jupyter-lab-script.py" in the Target

## Links
The instructions here are fairly complete and easy to follow:
https://towardsdatascience.com/setting-up-tensorflow-gpu-with-cuda-and-anaconda-onwindows-2ee9c39b5c44

This article has some more info and some code to test the install:
https://yann-leguilly.gitlab.io/post/2019-10-08-tensorflow-and-cuda/


