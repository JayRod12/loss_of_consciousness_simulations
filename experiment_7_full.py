
from brian2 import *
from echo_time import *
from neuron_groups import *

import sys
import math
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import power_spectral_density as psd

def run_experiment(n, w1, w2, w3, w4):#, duration, connectivity, scaling_factor, verbose=False):

    seed(152345)
    #seed(1777231)

    CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
    XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
    N = min(n, len(XYZ))
    
    EX_PROP = 0.8
    is_excitatory = np.array([np.random.random_sample() < EX_PROP for _ in range(N)])
    EX_NEURONS = np.arange(N)[is_excitatory]
    IN_NEURONS = np.arange(N)[~is_excitatory]
    
    N_EX = EX_NEURONS.size
    N_IN = N - N_EX 
    
    EX_G = ExcitatoryNeuronGroup(N_EX) 
    IN_G = InhibitoryNeuronGroup(N_IN)
    
    # PARAMS
    
    EX_EX_SCALING = w1
    EX_IN_SCALING = w2
    IN_EX_SCALING = w3
    IN_IN_SCALING = w4

    #EX_EX_SCALING = 150
    #EX_IN_SCALING = 50
    #IN_EX_SCALING = 4
    #IN_IN_SCALING = 4
    
    EX_EX_SCALING_DELAY = 4
    
    EX_EX_WEIGHT = 1*mV
    IN_EX_MIN_WEIGHT = -1*mV
    EX_IN_MAX_WEIGHT = 1*mV
    IN_IN_MIN_WEIGHT = -1*mV
    
    
    # Synapses
    echo = echo_start("Setting up Excitatory-Excitatory synapses... ")

    EX_EX_SYN = Synapses(EX_G,
        model='w : volt',
        on_pre='v += w',
 #       delay=1*ms
    )

    delay_matrix = np.zeros((N, N))
    synapses_i, synapses_j = [], []
    #delay_matrix = np.zeros((N_EX, N_EX))
    #for i in range(N_EX):
    #    ex_i = EX_NEURONS[i]
    #    x, y, z = XYZ[ex_i]
    #    for j in range(N_EX):
    #        ex_j = EX_NEURONS[j]
    #        if CIJ[ex_i][ex_j] > 0:
    #            x2, y2, z2 = XYZ[ex_j]
    #            delay_matrix[i][j] = math.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)/2
    #            #synapses_i.append(ex_i)
    #            #synapses_j.append(ex_j)
    #            synapses_i.append(i)
    #            synapses_j.append(j)

    for i in range(N):
        if i < N_EX:
            n_i = EX_NEURONS[i]
        else:
            n_i = IN_NEURONS[i-N_EX]
        x, y, z = XYZ[n_i]
        for j in range(N):
            if j < N_EX:
                n_j = EX_NEURONS[j]
            else:
                n_j = IN_NEURONS[j-N_EX]

            x2, y2, z2 = XYZ[n_j]
            delay_matrix[i][j] = math.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)/2
            if i < N_EX and j < N_EX and CIJ[n_i][n_j] > 0:
                synapses_i.append(i)
                synapses_j.append(j)
                

    EX_EX_SYN.connect(i=synapses_i, j=synapses_j)
    EX_EX_SYN.w = CIJ[EX_NEURONS[synapses_i], EX_NEURONS[synapses_j]] * EX_EX_SCALING * mV
    EX_EX_SYN.delay = delay_matrix[synapses_i, synapses_j] * ms
    echo_start(" ({:,} synapses)".format(len(synapses_i)))

    echo_end(echo)


    echo = echo_start("Setting up Excitatory-Inhibitory synapses... ")
    EX_IN_SYN = Synapses(EX_G, IN_G,
        model='w : volt',
        on_pre='v += w',
        delay=1*ms
    )
#    inhibitory_shuffled = np.random.permutation(N_IN)
    synapses_i = np.arange(N_EX)
    synapses_j = np.random.permutation(N_IN)[synapses_i % N_IN]
    EX_IN_SYN.connect(i=synapses_i, j=synapses_j)
#    EX_IN_SYN.delay = delay_matrix[synapses_i, synapses_j + N_EX] * ms

#    for i in range(N_EX):
#        EX_IN_SYN.connect(i=i, j=inhibitory_shuffled[i%N_IN])

    EX_IN_SYN.w[:,:] = rand() * EX_IN_SCALING * EX_IN_MAX_WEIGHT
    echo_start(" ({:,} synapses)".format(len(EX_IN_SYN.w)))

    echo_end(echo)
    
    echo = echo_start("Setting up Inhibitory-Excitatory synapses... ")
    IN_EX_SYN = Synapses(IN_G, EX_G,
        model='w : volt',
        on_pre='v += w',
        delay=1*ms
    )
    for in_neuron in range(N_IN):
        IN_EX_SYN.connect(i=in_neuron, j=range(N_EX))
    IN_EX_SYN.w[:,:] = rand() * IN_EX_SCALING * IN_EX_MIN_WEIGHT
    
    echo_start(" ({:,} synapses)".format(len(IN_EX_SYN.w)))
    echo_end(echo)
    
    echo = echo_start("Setting up Inhibitory-Inhibitory synapses... ")
    IN_IN_SYN = Synapses(IN_G,
        model='w : volt',
        on_pre='v += w',
        delay=1*ms
    )
    
    for in_neuron in range(N_IN):
        IN_IN_SYN.connect(i=in_neuron, j=np.arange(N_IN))
    
    IN_IN_SYN.w[:,:] = rand() * IN_IN_SCALING * IN_IN_MIN_WEIGHT
    echo_start(" ({:,} synapses)".format(len(IN_IN_SYN.w)))
    
    echo_end(echo) 
    
    # Poisson input to ensure network activity doesn't die down
    POISSON_INPUT_WEIGHT=2*mV
    PI_EX = PoissonInput(EX_G, 'v', len(EX_G), 1*Hz, weight=POISSON_INPUT_WEIGHT)
    #EX_G.v = -65*mV
    #IN_G.v = -65*mV
    
    M_EX = SpikeMonitor(EX_G)
    M_IN = SpikeMonitor(IN_G)
    
    
    echo = echo_start("Running simulation and plotting... ")
    
    # Monitors
    duration = 20000
    run(duration*ms)
    
    echo_end(echo)

    X, Y = np.array(M_EX.t/ms), np.array(M_EX.i)
    X2, Y2 = np.array(M_IN.t/ms), np.array(M_IN.i)
    DATA = {
        'duration': duration,
        'N': N,
        'N_EX': N_EX,
        'X': X,
        'Y': Y,
        'X2': X2,
        'Y2': Y2,
        #'M_EX': M_EX,
        #'M_IN': M_IN,
    }
    return DATA

