# Installation 

To install Anaconda on ubuntu 22.04 you can:

Anaconda (Python + R + pacakges, Conda env/package manager)
- Miniconda (Python + Conda env/package manager)
-- Conda (standalone env/package manager)

1. Get the Miniconda installer script from 
- https://docs.anaconda.com/free/miniconda/
-- https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
1. Run the script: `bash Miniconda3-latest-Linux-x86_64.sh`
2. Export the `conda/bin` to the default `PATH` variable
3. Run `conda init` to start default env

Base conda environment is now active on every shell you open.


# Creating a new environment

`conda create --name <ENV_NAME>`

## Exporting the environment config

To export the current config to a .yml file that can be used in other machines. You can use the following command:
`conda env export > conda_environment.yml`

## Importing the environment config
Use the `-f` and pass the file with the exported environment config.
`conda env create -f conda_environment.yml`


# Installing packages

To list every package available on the current chanels, use:
`conda search`


To install a specific package such as SciPy (can define the version too) into an existing environment "myenv":
`conda install --name myenv scipy=0.123`

Or simply use `conda install` with the environment active

# Uninstalling packages
`conda remove scipy`