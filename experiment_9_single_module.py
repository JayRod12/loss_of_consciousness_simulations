from brian2 import *
from neuron_groups import *
from izhikevich_constants import *
from numpy.random import random_sample
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
# Full-Ping oscillations >30 Hz

#np.random.seed(177876383)
#np.random.seed(177735)
np.random.seed(25)
DELAY = 5*ms

N_EX = 40
N_IN = 10

EX_CONNECTIVITY = 0.4
IN_CONNECTIVITY = 0.1

EX_G = ExcitatoryNeuronGroup(N_EX)
EX_G.I = 15*random_sample(N_EX)*mV/ms

IN_G = InhibitoryNeuronGroup(N_IN)
IN_G.I = 3*random_sample(N_IN)*mV/ms


# PARAMS
EX_EX_WEIGHT = 5*mV
EX_IN_WEIGHT = 10*mV
IN_EX_WEIGHT = -10*mV
IN_IN_WEIGHT = -10*mV

# 1-to-1 connections: EX_CONNECTIVITY% of all synapses
EX_EX_SYN = Synapses(EX_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
EX_EX_SYN.connect(p=EX_CONNECTIVITY)
EX_EX_SYN.w = EX_EX_WEIGHT


# Many-to-one connection: 4 excitatory to 1 inhibitory
EX_IN_SYN = Synapses(EX_G, IN_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
EX_IN_SYN.connect(j='int(i/4)')
EX_IN_SYN.w = EX_IN_WEIGHT


# All-to-all: diffuse inhibitory to excitatory connections
IN_EX_SYN = Synapses(IN_G, EX_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
IN_EX_SYN.connect(p=1.0)
IN_EX_SYN.w = IN_EX_WEIGHT


# 1-to-1: IN_CONNECTIVITY% of all synapses
IN_IN_SYN = Synapses(IN_G, IN_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
IN_IN_SYN.connect(p=IN_CONNECTIVITY)
IN_IN_SYN.w = IN_IN_WEIGHT

# Monitoring and simulation
M = SpikeMonitor(EX_G)
duration = 5000*ms
run(duration)

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

