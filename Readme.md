# Model for the _Drosophila_ brain
This repository accompanies the paper [A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1).


In this model, one can activate a set of [Flywire](flywire.ai) neurons at a particular frequency, then identify what neurons respond to this activation. Additionally, two sets of neurons (e.g., sugar and bitter sensing neurons) can be activated, and the interactions between these two neurons can be determined. Furthermore, one can also activate a set of neurons and silence a second set of neurons, and measure how this changes activity of a set of output neurons.


See the [example](https://github.com/philshiu/Drosophila_brain_model/blob/main/example.ipynb) file for details on how to perform these operations.
