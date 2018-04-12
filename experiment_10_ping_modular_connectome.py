
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

echo = echo_start("Setting up synapses... \n")

tt = time.time()
EX_EX_SYN.connect(
    condition='int(i/40) == int(j/40)',
    p=EX_CONNECTIVITY
)
print('\tEX_EX_SYN: {}s'.format(time.time() - tt))

tt = time.time()
EX_IN_SYN.connect(
    condition='j == int(i/4)'
)
print('\tEX_IN_SYN: {}s'.format(time.time() - tt))

tt = time.time()
IN_EX_SYN.connect(
    condition='int(i/10) == int(j/40)'
)
print('\tIN_EX_SYN: {}s'.format(time.time() - tt))

tt = time.time()
IN_IN_SYN.connect(
    condition='int(i/10) == int(j/10)',
    p=IN_CONNECTIVITY
)
print('\tIN_IN_SYN: {}s'.format(time.time() - tt))
    
EX_EX_SYN.w = EX_EX_WEIGHT
EX_IN_SYN.w = EX_IN_WEIGHT
IN_EX_SYN.w = IN_EX_WEIGHT
IN_IN_SYN.w = IN_IN_WEIGHT

echo_end(echo)

echo = echo_start("Running sym... ")
M = SpikeMonitor(EX_G)
duration = 20000*ms
run(duration)
echo_end(echo)


fname = "experiment_data/exp10_{}sec.pickle".format(duration/ms/1000)
echo = echo_start("Storing data to {}... ".format(fname))
DATA = {
    'X': np.array(M.t/ms),
    'Y': np.array(M.i),
    'duration': duration/ms
}

with open(fname, 'wb') as f:
    pickle.dump(DATA, f)

echo_end(echo)
