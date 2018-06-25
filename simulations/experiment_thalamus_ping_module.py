
"""
PING architecture module for the thalamus module.

This experiment sets up a module of neurons following the PING architecture to
achieve high-frequency oscillatory dynamics in the Gamma band (30-80Hz).
The parameters are different than the other ping module.
"""


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

#CONFIG = {
#    'n_ex': 40,
#    'n_in': 10,
#    'exin_w': (10,2),
#    'inex_w': (10,2),
#    'inin_w': (10,2),
#    'exin_d': (2,1),
#    'inex_d': (5,2),
#    'inin_d': (5,2)
#    'exin_conn': 0.7,
#    'inex_conn': 0.7,
#    'inin_conn': 0.7,
#
#}

def run_simulation(
        exex_w=(5,2),
        exin_w=(10,2),
        inex_w=(10,2),
        inin_w=(10,2),
        exex_d=(5,2),
        exin_d=(10,2),
        inex_d=(10,2),
        inin_d=(10,2)
    ):

    seed(1978331)

    n_ex = 200
    n_in = 50
    EX_G = ExcitatoryNeuronGroup(n_ex)
    IN_G = InhibitoryNeuronGroup(n_in)

    # PARAMS
    EX_EX_CONN = 0.3
    EX_IN_CONN = 0.4
    IN_EX_CONN = 0.4
    IN_IN_CONN = 0.4

    MIN_DELAY, MAX_DELAY= 1, 10
    MIN_W, MAX_W = 0, 15

    echo = echo_start("Setting up synapses:\n")

    EX_EX_SYN = Synapses(EX_G, EX_G,
        model='w: volt',
        on_pre='v += w',
    )
    echo2 = echo_start('\tEX_EX_SYN... ')

    EX_EX_SYN.connect(p=EX_EX_CONN)

    EX_EX_SYN.w[:,:] = np.clip(
        np.random.normal(exex_w[0], exex_w[1], size=len(EX_EX_SYN)),
        MIN_W, MAX_W
    ) * mV

    EX_EX_SYN.delay[:,:] = np.clip(
        np.random.normal(exex_d[0], exex_d[1], size=len(EX_EX_SYN)),
        MIN_DELAY, MAX_DELAY
    ) * ms
    echo_end(echo2, "({:,} synapses)".format(len(EX_EX_SYN)))

    EX_IN_SYN = Synapses(EX_G, IN_G,
        model='w: volt',
        on_pre='v += w',
    )
    echo2 = echo_start('\tEX_IN_SYN... ')

    EX_IN_SYN.connect(p=EX_IN_CONN)

    EX_IN_SYN.w[:,:] = np.clip(
        np.random.normal(exin_w[0], exin_w[1], size=len(EX_IN_SYN)),
        MIN_W, MAX_W
    ) * mV

    EX_IN_SYN.delay[:,:] = np.clip(
        np.random.normal(exin_d[0], exin_d[1], size=len(EX_IN_SYN)),
        MIN_DELAY, MAX_DELAY
    ) * ms
    echo_end(echo2, "({:,} synapses)".format(len(EX_IN_SYN)))

    # Inhibitory to excitatory connections
    IN_EX_SYN = Synapses(IN_G, EX_G,
        model='w: volt',
        on_pre='v -= w',
    )
    echo2 = echo_start('\tIN_EX_SYN... ')
    IN_EX_SYN.connect(p=IN_EX_CONN)

    IN_EX_SYN.w[:,:] = np.clip(
        np.random.normal(inex_w[0], inex_w[1], size=len(IN_EX_SYN)),
        MIN_W, MAX_W
    ) * mV

    IN_EX_SYN.delay[:,:] = np.clip(
        np.random.normal(inex_d[0], inex_d[1], size=len(IN_EX_SYN)),
        MIN_DELAY, MAX_DELAY
    ) * ms
    echo_end(echo2, "({:,} synapses)".format(len(IN_EX_SYN)))


    IN_IN_SYN = Synapses(IN_G, IN_G,
        model='w: volt',
        on_pre='v -= w',
    )
    echo2 = echo_start('\tIN_IN_SYN... ')

    IN_IN_SYN.connect(p=IN_IN_CONN)

    IN_IN_SYN.w[:,:] = np.clip(
        np.random.normal(inin_w[0], inin_w[1], size=len(IN_IN_SYN)),
        MIN_W, MAX_W
    ) * mV

    IN_IN_SYN.delay[:,:] = np.clip(
        np.random.normal(inin_d[0], inin_d[1], size=len(IN_IN_SYN)),
        MIN_DELAY, MAX_DELAY
    ) * ms

    echo_end(echo2, "({:,} synapses)".format(len(IN_IN_SYN)))

    POISSON_INPUT_WEIGHT=8*mV
    PI_EX = PoissonInput(EX_G, 'v', len(EX_G), 4.375*Hz, weight=POISSON_INPUT_WEIGHT)

    echo_end(echo)

    # Monitoring and simulation
    echo = echo_start("Running sym... ")
    M_EX = SpikeMonitor(EX_G)
    M_IN = SpikeMonitor(IN_G)
    duration = 5000 * ms
    run(duration)
    echo_end(echo)


    X, Y = np.array(M_EX.t / ms), np.array(M_EX.i)
    X_IN, Y_IN = np.array(M_IN.t / ms), np.array(M_IN.i)

    RESULTS = {
        'n': n_ex + n_in,
        'n_ex': n_ex,
        'n_in': n_in,
        'X': X,
        'Y': Y,
        'X2': X_IN,
        'Y2': Y_IN,
    }
    return RESULTS


