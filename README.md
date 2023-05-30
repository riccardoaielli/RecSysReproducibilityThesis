# SIGIR 2023 - Reproducibility of Recommendation Systems based on message passing

Developed by Anonymized for blind review.
All the papers we attempt to reproduce are in SIGIR2022 folder. For each algorithm there are two folders:
* A folder named "_github" which contains the full original repository, with the minor fixes needed for the code to run.
* A folder named "_our_interface" which contains the python wrappers needed to allow its testing in our framework. The main class for that algorithm has the "Wrapper" suffix in its name. This folder also contains the functions needed to read and split the data in the appropriate way.

The repository also contains a pdf with the full results, hyperparameter configuration and search space for the baselines.

## Run the experiments 

See the following [Installation](#Installation) section for information on how to install this repository.
After the installation is complete you can run the experiments.

### Comparison with baselines algorithms

This repository contains one script for each of the 9 papers we attempt to reproduce. Each script starts with _run__, the conference name and the year of publication.
The scripts have the following boolean optional parameters (all default values are False except for the print-results flag):
* '--baseline_tune': Run baseline hyperparameter search
* '--article_default': Train the algorithm with the original hyperparameters
* '--article_tune': Run hyperparameter search for the graph algorithm (available only for GDE, RGCF)
* '--print_results': Generate the latex tables for this experiment


For example, if you want to run all the experiments for LightGCN, you should run this command:
```console
python run_SIGIR20_LightGCN_experiment.py --baseline_tune==True --article_default==True --print_results==True
```


## Installation

Note that this repository requires Python 3.8

First we suggest you create an environment for this project using virtualenv (or another tool like conda)

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment, if you are using conda:
```console
conda create -n SIGIRReproducibility python=3.8 anaconda
conda activate SIGIRReproducibility
```

In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```console
sudo apt install gcc 
sudo apt-get install python3-dev
```

Then install all the requirements and dependencies
```console
pip install -r requirements.txt
```

In order to install torch you may either rely on conda or use pip. We use version 1.12.1 with cuda 11.6 
```console
pip install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install torch-geometric
```

At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```console
python run_compile_all_cython.py
```

