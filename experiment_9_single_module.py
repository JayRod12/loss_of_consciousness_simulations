from brian2 import *
from izhikevich_constants import *
from collections import defaultdict

import sys
import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
# Import after brian2, otherwise the variable psd is overwritten
import power_spectral_density as psd

# Model a single Brodmann area as 50 neurons
# Ping oscillations 35 Hz

# Network structure:
#  Excitatory -> Inhibitory
#     ^   |
#     |   |
#     -----

# Configuration options:
#  - Each excitatory neuron stimulates one inhibitory neuron which also
#       stimulates it back: doesn't work well.
#  - Excitatory-inhibitory and opposite connections are random and not necessarily bidirectional:
#       there isn't enough syncronization throughout the module, not good enough.
#  - Excitatory-inhibitory connections random and focal; Inhibitory-Excitatory conns are
#       diffuse, one-to-all: good option

np.random.seed(177876383)
DELAY = 5*ms

N_EX = 40
N_IN = 10

EX_CONNECTIVITY = 0.6

EX_G = NeuronGroup(N_EX,
    EXCITATORY_NEURON_EQS,
    threshold=THRES_EQ,
    reset=EXCITATORY_RESET_EQ,
    method='rk4'
)

IN_G = NeuronGroup(N_IN,
    INHIBITORY_NEURON_EQS,
    threshold=THRES_EQ,
    reset=INHIBITORY_RESET_EQ,
    method='rk4'
)


# PARAMS
EX_EX_WEIGHT = 4*mV
EX_IN_WEIGHT = 10*mV
IN_EX_WEIGHT = -10*mV
EX_EX_VARIANCE = 1*mV
EX_IN_VARIANCE = 2*mV
IN_EX_VARIANCE = 2*mV


EX_EX_SYN = Synapses(EX_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
EX_EX_CONNECTION = [(i, j) for i in range(N_EX) for j in range(N_EX) if random() <= EX_CONNECTIVITY]
tmp_i, tmp_j = map(list, zip(*EX_EX_CONNECTION))
EX_EX_SYN.connect(i=tmp_i, j=tmp_j)
EX_EX_SYN.connect(i=list(range(N_EX)), j=list(range(N_EX)))
EX_EX_SYN.w[:,:] = 'EX_EX_WEIGHT + EX_EX_VARIANCE*(2*rand() - 1)'


# Connect excitatory neurons to inhibitory neurons like so:
# 0-9 -> 0
# 10-19 -> 1
# 20-29 -> 2
# 30-39 -> 3
EX_IN_SYN = Synapses(EX_G, IN_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
EX_NEURONS = np.array(range(N_EX), dtype=int32)
EX_IN_SYN.connect(i=EX_NEURONS, j=EX_NEURONS/4)
EX_IN_SYN.w[:,:] = 'EX_IN_WEIGHT + EX_IN_VARIANCE*(2*rand() - 1)'


IN_EX_SYN = Synapses(IN_G, EX_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
IN_NEURONS = np.array(range(N_IN), dtype=int32)
for in_neuron in range(N_IN):
    IN_EX_SYN.connect(i=in_neuron, j=EX_NEURONS)
IN_EX_SYN.w[:,:] = 'IN_EX_WEIGHT + IN_EX_VARIANCE*(2*rand() - 1)'


POISSON_INPUT_WEIGHT=5*mV
PI_EX = PoissonInput(EX_G, 'v', len(EX_G), 30*Hz, weight=POISSON_INPUT_WEIGHT)




M = SpikeMonitor(EX_G)
duration = 5000*ms
run(duration)

plot_time = 500
spike_t, spike_idx = M.t/ms, M.i

MEASURE_START = 1000
MEASURE_DURATION = 500

X1, Y1 = [], []
for spike_t, spike_idx in zip(M.t/ms, M.i):
    if MEASURE_START <= spike_t < MEASURE_START + MEASURE_DURATION:
        X1.append(spike_t)
        Y1.append(spike_idx)



X, Y = M.t/ms, M.i

dt = 10 # ms
shift = 5 # ms
total_steps = int(duration/(shift*ms))
ma, time_scale = psd.moving_average(X, dt, shift, total_steps, True)

X2, Y2 = [], []
for ma_val, t in zip(ma, time_scale):
    if MEASURE_START <= t[0] < MEASURE_START + MEASURE_DURATION:
        Y2.append(ma_val)
        X2.append(t[0])


plt.subplot(211)
plt.plot(X1, Y1, '.b')
plt.ylabel('Neuron Index')
plt.xlabel('Simulation Time (ms)')

plt.subplot(212)
plt.plot(X2, Y2)
plt.xlabel('Simulation Time (ms)')
plt.ylabel('Mean firing rate')

plt.show()

