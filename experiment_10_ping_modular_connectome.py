
from brian2 import *
from neuron_groups import *
from echo_time import *

import pickle
import scipy.io as spio
import matplotlib.pyplot as plt
import power_spectral_density as psd

CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
N_MOD = len(XYZ)        # Number of modules

# Parameters

EX_EX_WEIGHT = 5*mV
EX_IN_WEIGHT = 10*mV
IN_EX_WEIGHT = -10*mV
IN_IN_WEIGHT = -10*mV

DELAY = 5*ms

EX_CONNECTIVITY = 0.4
IN_CONNECTIVITY = 0.1

# Setup
N_EX_MOD, N_IN_MOD = 40, 10
N_EX = N_EX_MOD * N_MOD
N_IN = N_IN_MOD * N_MOD

EX_G = ExcitatoryNeuronGroup(N_EX)
EX_G.I = 15*random_sample(N_EX)*mV/ms

IN_G = InhibitoryNeuronGroup(N_IN)
IN_G.I = 3*random_sample(N_IN)*mV/ms


# Define all synapse objects
EX_EX_SYN = Synapses(EX_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
EX_IN_SYN = Synapses(EX_G, IN_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
IN_EX_SYN = Synapses(IN_G, EX_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
IN_IN_SYN = Synapses(IN_G, IN_G,
    model='w: volt',
    on_pre='v += w',
    delay=DELAY
)
INTER_EX_EX_SYN = Synapses(EX_G, EX_G,
    model='w: volt',
    on_pre='v += w',
)

echo = echo_start("Setting up synapses... \n")

tt = time.time()
EX_EX_SYN.connect(
    condition='int(i/40) == int(j/40)',
    p=EX_CONNECTIVITY
)
EX_EX_SYN.w = EX_EX_WEIGHT
print('\tEX_EX_SYN ({:,} synapses): {}s'.format(len(EX_EX_SYN.w), time.time() - tt))

tt = time.time()
EX_IN_SYN.connect(
    condition='j == int(i/4)'
)
EX_IN_SYN.w = EX_IN_WEIGHT
print('\tEX_IN_SYN ({:,} synapses): {}s'.format(len(EX_IN_SYN.w), time.time() - tt))

tt = time.time()
IN_EX_SYN.connect(
    condition='int(i/10) == int(j/40)'
)
IN_EX_SYN.w = IN_EX_WEIGHT
print('\tIN_EX_SYN ({:,} synapses): {}s'.format(len(IN_EX_SYN.w), time.time() - tt))

tt = time.time()
IN_IN_SYN.connect(
    condition='int(i/10) == int(j/10)',
    p=IN_CONNECTIVITY
)
IN_IN_SYN.w = IN_IN_WEIGHT
print('\tIN_IN_SYN ({:,} synapses): {}s'.format(len(IN_IN_SYN.w), time.time() - tt))

# Synapses between modules (only excitatory-excitatory connections will be
# created)
INTER_MODULE_CONNECTIVITY = 0.1

tt = time.time()

synapses = []
delay_matrix = np.zeros((N_MOD, N_MOD))
for i in range(N_MOD):
    x, y, z = XYZ[i]
    for j in range(N_MOD):
        if CIJ[i][j] > 0:
            # Delay = distance / speed, speed = 2 m/s
            x2, y2, z2 = XYZ[j]
            delay_matrix[i][j] = math.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)/2
            base_i, base_j = i * N_EX_MOD, j * N_EX_MOD
            synapses += [(base_i + ii, base_j + jj)
                for ii in range(N_EX_MOD)
                for jj in range(N_EX_MOD)
                if sample() < INTER_MODULE_CONNECTIVITY
            ]

synapses_i, synapses_j = map(np.array, zip(*synapses))
INTER_EX_EX_SYN.connect(i=synapses_i, j=synapses_j)

INTER_EX_EX_SYN.delay = \
        delay_matrix[np.array(synapses_i/N_EX_MOD), np.array(synapses_j/N_EX_MOD)] * ms

INTER_EX_EX_SYN.w = CIJ[synapses_i/N_EX_MOD, synapses_j/N_EX_MOD] * mV

print('\tINTER_EX_EX_SYN ({:,} synapses): {}s'.format(len(INTER_EX_EX_SYN.w), time.time() - tt))

echo_end(echo)

echo = echo_start("Running sym... ")
M = SpikeMonitor(EX_G)
duration = 20000*ms
run(duration)
echo_end(echo)


fname = "experiment_data/exp10_{}sec.pickle".format(int(duration/ms/1000))
echo = echo_start("Storing data to {}... ".format(fname))
DATA = {
    'X': np.array(M.t/ms),
    'Y': np.array(M.i),
    'duration': duration/ms,
    'N_MOD': N_MOD
}

with open(fname, 'wb') as f:
    pickle.dump(DATA, f)

echo_end(echo)
