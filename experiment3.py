from watts_strogatz import watts_strogatz
from small_world_index import small_world_index
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_constants import *

from brian2 import *



# Small network of 100 neurons
n = 100
duration = 1000*ms
p = 0
(G, S, cij) = watts_strogatz(n, 4, p, EXCITATORY_NEURON_EQS, THRES_EQ,
        EXCITATORY_RESET_EQ, on_pre_action='v += 5*mV')

# Spike of a single neuron
inp = SpikeGeneratorGroup(1, [0], [0*ms])
feedforward = Synapses(inp, G, on_pre='v += weight')

spikemon = SpikeMonitor(G)
run(duration)

plt.plot(spikemon.t/ms, spikemon.i, '.b')
plt.show()




