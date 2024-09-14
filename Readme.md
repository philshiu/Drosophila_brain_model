[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philshiu/Drosophila_brain_model/blob/main/example.ipynb)

# Model for the _Drosophila_ brain
Activate and silence neurons in a computational model based on the
fruit fly connectome

# Paper
This repository accompanies the paper 
[A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1).
It includes all code necessary to reproduce the data presented there.
[figures.ipynb](figures.ipynb) contains the code used to perform the calculations.
However, the raw output of the model is several GB and therefore not posted on github.
Instead, it can be found in this [online archive](https://doi.org/10.17617/3.CZODIW).


# Usage
With this computational model,
one can manipulate the neural activity of a set of _Drosophila_ neurons, which can be addressed via their [Flywire](https://flywire.ai/) ID.
The output of the model is the spike times and rates of all affected neurons.

Two types of manipulations are currently implemented:
- *Activation*:
Neurons can be activated at a fixed frequency to model optogenetic activation.
This triggers Poisson spiking in the target neurons. 
Two sets of neurons with distinct frequencies can be defined.
- *Silencing*:
In addition to activation, a different set of neurons can be silenced to model optogenetic silencing.
This sets all synaptic connections to and from those neurons to zero.

See [example.ipynb](example.ipynb) for a tutorial and a more detailed explanation
The actual model resides in [model.py](model.py).
Some helper functions are provided by [utils.py](utils.py).

# Installation
The Drosophila brain model can be run on Mac, Windows or Unix, and installation should take 10 minutes.
## Quick Start
To begin using the model without a local install, click on the _Open In Colab_ badge at the top of this README. Note that simulations can take substantially longer to run on Google Colab than on a local installation, depending on the number of CPU cores you have access to.
## via conda
Install in a [Anaconda](https://www.anaconda.com/) environment:
```bash
conda env create -f environment.yml
```
Run using jupyter notebooks as shown in [example.ipynb](example.ipynb) and [figures.ipynb](figures.ipynb).

Note that this may install *Brian 2* with limited performance (see below).

## Brian 2 performance
The model is written in python built using the *Brian 2* simulator.
See the official [Brian 2 documentation](https://brian2.readthedocs.io/en/stable/introduction/install.html) for detailed installation instructions for you system.
Specifically, follow the instructions for [C++ code generation](https://brian2.readthedocs.io/en/stable/introduction/install.html#requirements-for-c-code-generation) to install *Brian 2* with increased performance.

## dependencies
See [environment_full.yml](environment_full.yml) for specific package versions used in the original work.
## Version 783
The code is currently set up for the Flywire version 630, which the paper used. To use public version 783, change the config dictionary:
```python
config = {
    'path_res'  : './results/new',                 # directory to store results
    'path_comp' : './Completeness_783.csv',        # csv of the complete list of Flywire neurons
    'path_con'  : './Connectivity_783.parquet',    # connectivity data
    'n_proc'    : -1,                              # number of CPU cores (-1: use all)
}
