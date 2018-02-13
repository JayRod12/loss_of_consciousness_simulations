from watts_strogatz import watts_strogatz
from small_world_index import small_world_index
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_constants import *

from brian2 import *

# Experiment 2: Testing the synapse weight


duration = 1000*ms
N = 2
G = NeuronGroup(N,
        EXCITATORY_NEURON_EQS,
        threshold=THRES_EQ,
        reset=EXCITATORY_RESET_EQ,
        method='rk4')

I0 = 4
G.I = list([I0]*N)*mV/ms
G.I[0] = 0*mV/ms

w = 16*mV
S = Synapses(G, on_pre='v += w')
S.connect(i=np.arange(1,N), j=0)

M = StateMonitor(G, 'v', record=True)
spikemon = SpikeMonitor(G)
run(duration)


plt.subplot(311)
plt.plot(M.t/ms, M.v[0])
plt.title('Neuron 0 - Voltage')

plt.subplot(312)
plt.plot(M.t/ms, M.v[1])
plt.title('Neuron 1 - Voltage')

plt.subplot(313)
plt.plot(spikemon.t/ms, spikemon.i, '.b')
plt.title('Neuron spikes')

plt.tight_layout()
plt.show()

