
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_constants import *
from brian2 import *
from visualise import visualise_connectivity


EX_EX_WEIGHT = 1*mV
IN_EX_MIN_WEIGHT = -1*mV
EX_IN_MAX_WEIGHT = 1*mV
IN_IN_MIN_WEIGHT = -1*mV

EX_EX_SCALING = 17
EX_IN_SCALING = 50
IN_EX_SCALING = 2
IN_IN_SCALING = 1

EX_EX_MAX_DELAY = 20*ms

N_MODULES = 8
N_EXCITATORY = 100
N_INTERNAL_CONN = 1000
N_INHIBITORY = 200

# Number of excitatory neurons connecting to a single inhibitory neuron
FOCALITY = 4

def create_excitatory_neurons(N, N_CONN, N_MODULES):
    G = NeuronGroup(N*N_MODULES,
            EXCITATORY_NEURON_EQS,
            threshold=THRES_EQ,
            reset=EXCITATORY_RESET_EQ,
            method='rk4')
    S = Synapses(G, on_pre='v += EX_EX_SCALING * EX_EX_WEIGHT')
    for module in range(N_MODULES):
        i = np.random.randint(N, size=N_CONN)+N*module
        j = np.random.randint(N, size=N_CONN)+N*module
        S.connect(i=i, j=j)

    S.delay[:,:] = 'rand()*EX_EX_MAX_DELAY'
    return G, S

def create_inhibitory_neurons(N):
    G = NeuronGroup(N,
            INHIBITORY_NEURON_EQS,
            threshold=THRES_EQ,
            reset=INHIBITORY_RESET_EQ,
            method='rk4')
    return G


# N_MODULES modules of N_EXCITATORY excitatory neurons each, with
# N_INTERNAL_CONN internal random connections.
EX_NEURONS, EX_EX_SYN = create_excitatory_neurons(N_EXCITATORY, N_INTERNAL_CONN, N_MODULES)
INHIBITORY_NEURONS = create_inhibitory_neurons(N_INHIBITORY)


# Focal excitatory-inhibitory connections

# 4 excitatory neurons (from the same module) project to each inhibitory neuron
# Select 4 successively-numbered neurons from each module to map it to an
# inhibitory neuron.
EX_IN_SYN = Synapses(EX_NEURONS,
    INHIBITORY_NEURONS,
    on_pre='v += EX_IN_SCALING * rand() * EX_IN_MAX_WEIGHT',
    delay=1*ms
)

perm = np.random.permutation(N_INHIBITORY)
for neuron_group in perm:
    i = [neuron_group*FOCALITY+k for k in range(FOCALITY)]
    EX_IN_SYN.connect(i=i, j=neuron_group)

# Diffuse inhibitory-excitatory connections
IN_EX_SYN = Synapses(INHIBITORY_NEURONS,
    EX_NEURONS,
    on_pre='v += IN_EX_SCALING * rand() * IN_EX_MIN_WEIGHT',
    delay=1*ms
)
for i in range(N_INHIBITORY):
    IN_EX_SYN.connect(i=i, j=range(N_MODULES*N_EXCITATORY))



# Diffuse inhibitory-inhibitory connections
IN_IN_SYN = Synapses(INHIBITORY_NEURONS, INHIBITORY_NEURONS,
        on_pre = 'v += IN_IN_SCALING * rand() * IN_IN_MIN_WEIGHT',
        delay=1*ms)

for i in range(N_INHIBITORY):
    IN_IN_SYN.connect(i=i, j=range(N_INHIBITORY))


POISSON_INPUT_WEIGHT=10*mV
PI_EX = PoissonInput(EX_NEURONS, 'v', len(EX_NEURONS), 1*Hz, weight=POISSON_INPUT_WEIGHT)

M = SpikeMonitor(EX_NEURONS)
M2 = SpikeMonitor(INHIBITORY_NEURONS)

# Monitors
duration = 1000*ms
run(duration)

plt.subplot(211)
plt.plot(M.t/ms, M.i, '.b') 
plt.subplot(212)
plt.plot(M2.t/ms, M2.i, '.k') 
plt.show()
    


