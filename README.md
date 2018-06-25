# Neural Simulations of Loss of Consciousness

This repository contains code associated to a series of brain simulations with
spiking neuron models. These experiments are attempting to reproduce the effects
that changes in the level of consciousness have on the dynamics of the brain.

The relevant experiments are located in the ``simulations/`` folder. We present each below.

## Experiments

### 1. PING Connectome Experiment
Found in ``simulations/experiment_connectome_ping.py``.
This experiment combines PING oscillators in every module of the the connectome.

It provides optional flags for also simulating the Thalamus and/or a TMS pulse.

A sample run of this code would be as follows.
```python
import models.experiment_connectome_ping as ex
from utils.plotlib import *

data = ex.run_simulation(n_mod=100, duration=5000, with_tms=False, with_thalamus=True)
plot_sim(data, max_mod=10)
```

All the below experiments have the same way of running them. For details about the parameters
used in each experiment, see the relevant files.

### 2. Single Neuron Connectome Experiment
Found in ``simulations/experiment_connectome_single_neurons.py``.
This connectome experiment places single neurons on every module of the simulation.

### 3. PING Module Experiment
Found in ``simulations/experiment_ping_module.py``.
This experiment consisted on investigating the range of possible dynamics of a PING module, and tuning the synaptic weights and delays until good oscillatory behaviour was reached.

### 4. Thalamus PING Module
Found in ``simulations/experiment_thalamus_ping_module.py``. Similar to the previous experiment, the Thalamus experiment simply consisted on taking the PING simulation and modifying some parameters to achieve different dynamics. To test the Thalamus in combination with the rest of the connectome, refer to the first experiment.

### Other experiments
This repository also contains other experiments done along the way, which we provide undocumented for reference and for anyone interested. They are found in the ``additional_simulations/`` folder.

## Notebooks

The ``sim_notebooks/`` folder contains the Jupyter notebooks where all the graphs and the above experiments were simulated. They are not commented and well structured, but they serve simply as reference and proof of all the work we have done. If the experiments want to be reproduced, we recommend using the code of the four experiments in a new Jupyter Notebook and following the approaches described in the report and in this document.

Other notebooks which are not as core to this investigation are provided in the folder ``additional_notebooks/``

## Utils
The ``utils/`` folder contains methods used throughout the notebooks to plot and calculate different measures on the output of the simulations.

Files of relevance are:
* ``lz76.py``: Calculates Lempel-Ziv complexity on the data
* ``pci.py``: Approximation of the PCI index of consciousness.
* ``power_spectral_density.py``: Calculates moving average firing rate and power-frequency spectrum of simulation data.
* ``neuron_groups.py``: Contains methods for creating populations of excitatory and inhibitory populations of Izhikevich neurons.