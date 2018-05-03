from brian2 import *
from echo_time import *
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

def run_simulation(ex_current, in_current, ex_conn=0.4, in_conn=0.1):
    DELAY = 5*ms

    N_EX = 40
    N_IN = 10

    EX_CONNECTIVITY = ex_conn
    IN_CONNECTIVITY = in_conn

    EX_G = ExcitatoryNeuronGroup(N_EX)
    EX_G.I = ex_current*random_sample(N_EX)*mV/ms

    IN_G = InhibitoryNeuronGroup(N_IN)
    IN_G.I = in_current*random_sample(N_IN)*mV/ms


    # PARAMS
    EX_EX_WEIGHT = 5*mV
    EX_IN_WEIGHT = 10*mV
    IN_EX_WEIGHT = -10*mV
    IN_IN_WEIGHT = -10*mV

    echo = echo_start("Setting up synapses... \n")

    # 1-to-1 connections: EX_CONNECTIVITY% of all synapses
    EX_EX_SYN = Synapses(EX_G,
        model='w: volt',
        on_pre='v += w',
        delay=DELAY
    )
    tt = time.time()
    EX_EX_SYN.connect(p=EX_CONNECTIVITY)
    EX_EX_SYN.w = EX_EX_WEIGHT
    print('\tEX_EX_SYN ({:,} synapses): {}s'.format(len(EX_EX_SYN.w), time.time() - tt))


    # Many-to-one connection: 4 excitatory to 1 inhibitory
    EX_IN_SYN = Synapses(EX_G, IN_G,
        model='w: volt',
        on_pre='v += w',
        delay=DELAY
    )
    tt = time.time()
    EX_IN_SYN.connect(j='int(i/4)')
    EX_IN_SYN.w = EX_IN_WEIGHT
    print('\tEX_IN_SYN ({:,} synapses): {}s'.format(len(EX_IN_SYN.w), time.time() - tt))


    # All-to-all: diffuse inhibitory to excitatory connections
    IN_EX_SYN = Synapses(IN_G, EX_G,
        model='w: volt',
        on_pre='v += w',
        delay=DELAY
    )
    tt = time.time()
    IN_EX_SYN.connect(p=1.0)
    IN_EX_SYN.w = IN_EX_WEIGHT
    print('\tIN_EX_SYN ({:,} synapses): {}s'.format(len(IN_EX_SYN.w), time.time() - tt))


    # 1-to-1: IN_CONNECTIVITY% of all synapses
    IN_IN_SYN = Synapses(IN_G, IN_G,
        model='w: volt',
        on_pre='v += w',
        delay=DELAY
    )
    tt = time.time()
    IN_IN_SYN.connect(p=IN_CONNECTIVITY)
    IN_IN_SYN.w = IN_IN_WEIGHT
    print('\tIN_IN_SYN ({:,} synapses): {}s'.format(len(IN_IN_SYN.w), time.time() - tt))

    echo_end(echo)

    # Monitoring and simulation
    echo = echo_start("Running sym... ")
    M_EX = SpikeMonitor(EX_G)
    M_IN = SpikeMonitor(IN_G)
    M_V = StateMonitor(EX_G, 'v', record=[0,1])
    duration = 5000*ms
    run(duration)
    echo_end(echo)


    start_time = 1000
    end_time = 1500

    echo = echo_start("Processing data and plotting... ")
    X, Y = M_EX.t/ms, M_EX.i
    mask = np.logical_and(X >= start_time, X < end_time)
    X, Y = X[mask], Y[mask]
    dt, shift = 10, 5 # ms
    ma, time_scale = psd.moving_average(X, dt, shift, start_time, end_time)

    plt.subplot(221)
    plt.plot(X, Y, '.b')
    plt.xlabel('Simulation Time')
    plt.ylabel('Neuron Index (Ex)')

    plt.subplot(222)
    plt.plot(time_scale, ma)
    plt.xlabel('Simulation Time (ms)')
    plt.ylabel('Mean firing rate (Ex)')

    X_IN, Y_IN = M_IN.t/ms, M_IN.i
    mask = np.logical_and(X_IN >= start_time, X_IN < end_time)
    X_IN, Y_IN = X_IN[mask], Y_IN[mask]
    dt, shift = 10, 5 # ms
    ex_ma, ex_time_scale = psd.moving_average(X_IN, dt, shift, start_time, end_time)

    plt.subplot(223)
    plt.plot(X_IN, Y_IN, '.b')
    plt.xlabel('Simulation Time (ms) (In)')
    plt.ylabel('Neuron Index (In)')

    plt.subplot(224)
    plt.plot(ex_time_scale, ex_ma)
    plt.xlabel('Simulation Time (ms)')
    plt.ylabel('Mean firing rate (In)')

    plt.show()

    echo_end(echo)
    return (X, Y, X_IN, Y_IN, (M_V.t/ms, M_V.v[0]))

if __name__ == '__main__':
    ex_current = 15
    in_current = 3
    run_experiment(ex_current, in_current)

