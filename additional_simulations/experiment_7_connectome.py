import math
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from izhikevich_constants import *
import power_spectral_density as psd
import pickle
import sys

with open('precomputed_exp_7.pickle', 'rb') as f:
    config = pickle.load(f)

[N, N_EX, N_IN, EX_NEURONS, IN_NEURONS, WEIGHTS, DELAYS, CONNECTIONS] = config

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

EX_EX_SCALING = 300
EX_IN_SCALING = 50
IN_EX_SCALING = 4
IN_IN_SCALING = 4

EX_EX_SCALING_DELAY = 4

EX_EX_WEIGHT = 1*mV
IN_EX_MIN_WEIGHT = -1*mV
EX_IN_MAX_WEIGHT = 1*mV
IN_IN_MIN_WEIGHT = -1*mV


# Synapses
sys.stdout.write("Setting up Excitatory-Excitatory synapses... ")
sys.stdout.flush()

EX_EX_SYN = Synapses(EX_G,
    model='w : volt',
    on_pre='v += w'
)

for i in range(N_EX):
    if CONNECTIONS[i]:
        EX_EX_SYN.connect(i=i, j=CONNECTIONS[i])
        EX_EX_SYN.w[i,:] = WEIGHTS[i, CONNECTIONS[i]] * EX_EX_SCALING * mV
        EX_EX_SYN.delay[i,:] = DELAYS[i, CONNECTIONS[i]] / EX_EX_SCALING_DELAY * ms

sys.stdout.write("Done\n")
sys.stdout.flush()

sys.stdout.write("Setting up Excitatory-Inhibitory synapses... ")
sys.stdout.flush()
EX_IN_SYN = Synapses(EX_G, IN_G,
    model='w : volt',
    on_pre='v += w',
    delay=1*ms
)
inhibitory_shuffled = np.random.permutation(N_IN)
for i in range(N_EX):
    EX_IN_SYN.connect(i=i, j=inhibitory_shuffled[i%N_IN])
EX_IN_SYN.w[:,:] = rand() * EX_IN_SCALING * EX_IN_MAX_WEIGHT

sys.stdout.write("Done\n")
sys.stdout.flush()

sys.stdout.write("Setting up Inhibitory-Excitatory synapses... ")
sys.stdout.flush()
IN_EX_SYN = Synapses(IN_G, EX_G,
    model='w : volt',
    on_pre='v += w',
    delay=1*ms
)
for in_neuron in range(N_IN):
    IN_EX_SYN.connect(i=in_neuron, j=range(N_EX))
IN_EX_SYN.w[:,:] = rand() * IN_EX_SCALING * IN_EX_MIN_WEIGHT

sys.stdout.write("Done\n")
sys.stdout.flush()

sys.stdout.write("Setting up Inhibitory-Inhibitory synapses... ")
sys.stdout.flush()
IN_IN_SYN = Synapses(IN_G,
    model='w : volt',
    on_pre='v += w',
    delay=1*ms
)

for in_neuron in range(N_IN):
    IN_IN_SYN.connect(i=in_neuron, j=np.arange(N_IN))

IN_IN_SYN.w[:,:] = rand() * IN_IN_SCALING * IN_IN_MIN_WEIGHT

sys.stdout.write("Done\n")
sys.stdout.flush()


# Poisson input to ensure network activity doesn't die down
POISSON_INPUT_WEIGHT=2*mV
PI_EX = PoissonInput(EX_G, 'v', len(EX_G), 1*Hz, weight=POISSON_INPUT_WEIGHT)
EX_G.v = -65*mV
IN_G.v = -65*mV

M_EX = SpikeMonitor(EX_G)
M_IN = SpikeMonitor(IN_G)


print("Running simulation and plotting")

# Monitors
duration = 20000*ms
run(duration)


# Pickle out simulation data
PICKLE_OUT_DATA = [
    int(duration/ms),
    list(M_EX.t/ms),
    list(M_IN.t/ms),
]
#fname = 'experiment_data/exp7conn_{}_{}-{}-{}-{}.pickle'.format(
#    int(duration/ms),
#    EX_EX_SCALING,
#    EX_IN_SCALING,
#    IN_EX_SCALING,
#    IN_IN_SCALING,
# )
#
#
#with open(fname, 'wb') as f:
#    print('Writing output data to \'{}\''.format(fname))
#    pickle.dump(PICKLE_OUT_DATA, f)


# Discard initial transient signal
transient_thres = 1000 # ms
X = np.concatenate([M_EX.t/ms, M_IN.t/ms])
Y = np.concatenate([M_EX.i, M_IN.i+N_EX])

XY = [(t, s) for (t, s) in zip(X, Y) if t >= transient_thres]
X, Y = zip(*XY)

plt.figure(1)
plt.plot(X, Y, '.b')
plt.xlabel("Time (ms)")
plt.ylabel("Excitatory Neuron Index")


plt.figure(2)
dt = 75 # ms
shift = 10 # ms
total_steps = int(duration/(shift*ms))
f, pxx = psd.power_spectrum(X, dt, shift, int((duration/ms)/shift))
plt.semilogy(f, pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectrum')


plt.show()

