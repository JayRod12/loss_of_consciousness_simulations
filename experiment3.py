from watts_strogatz import watts_strogatz
from small_world_index import small_world_index
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_constants import *

from brian2 import *

# Watts-Strogatz network of 800 spiking excitatory neurons.
# Parameters: k = 6, p = 0.01 -> 0.5
# A single feedforward connection is used as the initial spiking going into
# the network.
# Synapse simulated as an increase of `weight` mV to the post synaptic neuron
# after a pre-synaptic neuron spikes.
# Experiment varies p to see how the wave of activation varies

# Problems: I don't see the wave of activation reactivate the network
# as I increase p, as the Comp. Neuro. slides suggest should happen.

# Small network of 100 neurons
n = 800
duration = 1000*ms
p = 0.01
weight = 20*mV
(G, S, cij) = watts_strogatz(n, 6, p, EXCITATORY_NEURON_EQS, THRES_EQ,
        EXCITATORY_RESET_EQ, on_pre_action='v += weight', delay=2*ms)
G.v = -65*mV

# Spike of a single neuron
inp = SpikeGeneratorGroup(1, [0], [100*ms])
feedforward = Synapses(inp, G, on_pre='v += weight')
feedforward.connect(i=0,j=0)

spikemon = SpikeMonitor(G)
inp_mon = SpikeMonitor(inp)
statemon = StateMonitor(G, 'v', record=[0])

run(duration)

plt.subplot(211)
plt.plot(spikemon.t/ms, spikemon.i, '.b')
plt.xlim(0, 1000)
plt.title('Neuron spikes')
plt.ylabel('Neuron index')
plt.xlabel('Time (ms)')

plt.subplot(212)
plt.plot(statemon.t/ms, statemon.v[0])
plt.xlim(0, 1000)
plt.title('Neuron 0 - Voltage')
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()




