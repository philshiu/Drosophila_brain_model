# Model for the _Drosophila_ brain
Activate and silence neurons in a computational model based on the
fruit fly connectome

# Paper
This repository accompanies the paper 
[A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1).
It includes all code necessary to reproduce the data presented there.
[figures.ipynb](figures.ipynb) contains the code used to perfrom the calculations.
However, the raw output of the model is several GB and therefore not posted on github.
Instead, it can be found in this [online archive](TODO).




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
## Quick start 
Install via [Anaconda](https://www.anaconda.com/):
```bash
conda create -f conda_env.yml
```
Run using jupyter notebooks as shown in [example.ipynb](example.ipynb) and [figures.ipynb](figures.ipynb).

## Manual installation
The model is written in python built using the *Brian 2* simulator.
See the official [Brian 2 documentation](https://brian2.readthedocs.io/en/stable/introduction/install.html) for detailed installation instructions.

Other dependencies are the python packages.
(see [conda_env_full.yml](conda_env_full.yml) for specific package versions):
```
brian2
joblib
pandas
pyarrow
```

