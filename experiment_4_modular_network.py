
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



def create_excitatory_module(N, N_CONN):
    G = NeuronGroup(N,
            EXCITATORY_NEURON_EQS,
            threshold=THRES_EQ,
            reset=EXCITATORY_RESET_EQ,
            method='rk4')
    S = Synapses(G, on_pre='v += EX_EX_SCALING * EX_EX_WEIGHT')
    S.connect(i=np.random.randint(N, size=N_CONN),
            j=np.random.randint(N, size=N_CONN))
    S.delay[:,:] = 'rand()*EX_EX_MAX_DELAY'
    #print(S.delay)
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
#EXCITATORY_MODULES = []
#for i in range(N_MODULES):
#    EXCITATORY_MODULES.append(
#        create_excitatory_module(N_EXCITATORY, N_INTERNAL_CONN)
#    )
EXCITATORY_MODULES = [
    create_excitatory_module(N_EXCITATORY, N_INTERNAL_CONN)
        for _ in range(N_MODULES)
]

INHIBITORY_NEURONS = create_inhibitory_neurons(N_INHIBITORY)


# Focal excitatory-inhibitory connections

# 4 excitatory neurons (from the same module) project to each inhibitory neuron
# Select 4 successively-numbered neurons from each module to map it to an
# inhibitory neuron.
EX_IN_SYN = []
perm = np.random.permutation(N_INHIBITORY)
for neuron_group in perm:
    module = neuron_group // (N_EXCITATORY / FOCALITY)
    neuron_index = neuron_group % (N_EXCITATORY / FOCALITY)
    S = Synapses(EXCITATORY_MODULES[module][0],
            INHIBITORY_NEURONS,
            on_pre='v += EX_IN_SCALING * rand() * EX_IN_MAX_WEIGHT',
            delay=1*ms)
    S.connect(i=[neuron_index + i for i in range(FOCALITY)],
            j=neuron_group)
    EX_IN_SYN.append(S)

# Diffuse inhibitory-excitatory connections
IN_EX_SYN = []
for module in range(N_MODULES):
    S = Synapses(INHIBITORY_NEURONS,
            EXCITATORY_MODULES[module][0],
            on_pre='v += IN_EX_SCALING * rand() * IN_EX_MIN_WEIGHT',
            delay=1*ms)

    for in_neuron in range(N_INHIBITORY):
        # One-to-all
        S.connect(i=in_neuron, j=range(N_EXCITATORY))
    IN_EX_SYN.append(S)


# Diffuse inhibitory-inhibitory connections
IN_IN_SYN = Synapses(INHIBITORY_NEURONS, INHIBITORY_NEURONS,
        on_pre = 'v += IN_IN_SCALING * rand() * IN_IN_MIN_WEIGHT',
        delay=1*ms)

for i in range(N_INHIBITORY):
    IN_IN_SYN.connect(i=i, j=range(N_INHIBITORY))



G1, G2, G3, G4, G5, G6, G7, G8 = map(lambda tup: tup[0], EXCITATORY_MODULES)
S1, S2, S3, S4, S5, S6, S7, S8 = map(lambda tup: tup[1], EXCITATORY_MODULES)
SS1, SS2, SS3, SS4, SS5, SS6, SS7, SS8 = IN_EX_SYN

NEURON_GROUPS = [G1, G2, G3, G4, G5, G6, G7, G8, INHIBITORY_NEURONS]

POISSON_INPUT_WEIGHT = 15*mV
I1, I2, I3, I4, I5, I6, I7, I8, I9 = [
    PoissonInput(G, 'v', len(G), 1*Hz, weight=POISSON_INPUT_WEIGHT) for 
        G in NEURON_GROUPS
]

# Monitors
M1, M2, M3, M4, M5, M6, M7, M8, M9 = list(map(lambda G: SpikeMonitor(G),
        [G1, G2, G3, G4, G5, G6, G7, G8, INHIBITORY_NEURONS]))

spikemons = [M1, M2, M3, M4, M5, M6, M7, M8, M9]
duration = 1000*ms
run(duration)

for i in range(N_MODULES+1):
    plt.subplot(N_MODULES+1,1,i+1)
    plt.plot(spikemons[i].t/ms, spikemons[i].i, '.b') 

plt.show()
    


