# Neural Simulations of Loss of Consciousness

Experimenting with Spiking Neural networks in Brian2

## Experiment 1
Create a Watts-Strogatz network and vary the rewiring probability while measuring the Small World Index.

## Experiment 2
Testing synaptic setups.

Basic synapse on_pre action 'v += weight'.

## Experiment 5
8 modules of excitatory neurons (100 neurons each) and 1000 random connections per module.
One module of inhibitory neurons.

As per experimental setup in Dynamical Complexity slides.

## Experiment 7

Connectome experiment.

Using a connectome with weights and distances between 998 Brodmann areas.

- Each area modelled as a single neuron.
- 80%  excitatory neurons, 20% inhibitory neurons.
- Izhikevich neuron model.
- Excitatory-Excitatory synapses based on the weights and delays of the connectome.
- Excitatory-Inhibitory synapses: Focal Many-to-one connections.
- Inhibitory-Excitatory synapses: Diffuse one-to-all connections.
- Inhibitory-Inhibitory synapses: Diffuse all-to-all

Scalings have been adjusted based on the resulting raster plots.


## Experiment 8

Connectome experiment

Use the same connectome with each area of brodmann as a module with 10-100 neurons
within each module. Inhibition is contained within these modules.

- Each Brodmann area modelled as a module with 10-100 neurons.



