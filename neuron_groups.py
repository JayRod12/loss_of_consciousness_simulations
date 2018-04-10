from brian2 import *
from izhikevich_constants import *

# Neuron groups with heterogeneity as presented in Simple Model of Spiking
# Neurons, IEEE Transactions on Neural Networks (2003) 14:1569-1572,
# Eugene Izhikevich.

def ExcitatoryNeuronGroup(N):
    G = NeuronGroup(N,
        EXCITATORY_NEURON_EQS,
        threshold=THRES_EQ,
        reset=EXCITATORY_RESET_EQ,
        method='rk4'
    )
    G.rnd = 'rand()'
    G.a = 0.02/ms
    G.b = 0.2/ms
    G.c = '(-65 + 15 * (rnd ** 2)) * mV'
    G.d = '(8 - 6 * (rnd ** 2)) * mV / ms'
    return G

def InhibitoryNeuronGroup(N):
    G = NeuronGroup(N,
        INHIBITORY_NEURON_EQS,
        threshold=THRES_EQ,
        reset=INHIBITORY_RESET_EQ,
        method='rk4'
    )
    G.rnd = 'rand()'
    G.a = '(0.02 + 0.08 * rnd) / ms'
    G.b = '(0.25 - 0.05 * rnd) / ms'
    G.c = -65 * mV
    G.d = 2 * mV / ms
    return G


