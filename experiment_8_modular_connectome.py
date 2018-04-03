from brian2 import *
from izhikevich_constants import *

import sys
import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import power_spectral_density as psd

# Areas of Brodmann used as modules of neurons
#
# The i'th module's excitatory neurons are found at [i * N_EX_PER_MOD, (i+1) * N_EX_PER_MOD)
# The i'th module's inhibitory neurons are found at [i * N_EX_PER_MOD, (i+1) * N_EX_PER_MOD)

with open('precomputed_exp_8.pickle', 'rb') as f:
    config = pickle.load(f)

[N, N_MOD, N_PER_MOD, N_EX, N_IN, N_EX_PER_MOD, N_IN_PER_MOD,
    WEIGHTS, DELAY, CONNECTIONS, INTERNAL_SYNAPSE_PROP, FOCALITY] = config

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

EX_EX_INTRA_SCALING = 17
EX_EX_INTER_SCALING = 300
EX_IN_SCALING = 50
IN_EX_SCALING = 2
IN_IN_SCALING = 1

EX_EX_SCALING_DELAY = 4

EX_EX_MAX_DELAY = 20*ms

EX_EX_WEIGHT = 1*mV
IN_EX_MIN_WEIGHT = -1*mV
EX_IN_MAX_WEIGHT = 1*mV
IN_IN_MIN_WEIGHT = -1*mV

# Synapses

start = time.time()
sys.stdout.write("Setting up intermodular excitatory-excitatory synapses... ")
sys.stdout.flush()

EX_EX_INTER_SYN = Synapses(EX_G,
    model='w : volt',
    on_pre='v += w'
)
N_INTRA_SYNAPSES = 4
# Improvements:
# How to connect two modules? How many synapses between two modules?
for i in range(N_MOD):
    for j in CONNECTIONS[i]:
        src = np.random.randint(N_EX_PER_MOD, size=N_INTRA_SYNAPSES) + i * N_EX_PER_MOD
        dst = np.random.randint(N_EX_PER_MOD, size=N_INTRA_SYNAPSES) + j * N_EX_PER_MOD

        EX_EX_INTER_SYN.connect(i=src, j=dst)
        EX_EX_INTER_SYN.w[src[:, None],dst] = min(WEIGHTS[i, j] * 1000, 10) * mV

elapsed = time.time() - start
sys.stdout.write("Done [{} seconds]\n".format(elapsed))
sys.stdout.flush()

exit()




sys.stdout.write("Setting up intramodular excitatory-excitatory synapses... ")
sys.stdout.flush()

EX_EX_SYN = Synapses(EX_G,
    model='w : volt',
    on_pre='v += w'
)

start_index = 0
max_synapses = N_EX_PER_MOD * N_EX_PER_MOD - 1
synapses_per_mod = int(INTERNAL_SYNAPSE_PROP * max_synapses)
for _ in range(N_MOD):
    # Connect neurons pairwise, i.e. (i[0], j[0]), (i[1], j[1]), ...
    i = np.random.randint(N_EX_PER_MOD, size=synapses_per_mod) + start_index
    j = np.random.randint(N_EX_PER_MOD, size=synapses_per_mod) + start_index

    EX_EX_SYN.connect(i=i, j=j)

    start_index += N_EX_PER_MOD

EX_EX_SYN.w[:,:] = EX_EX_INTRA_SCALING * EX_EX_WEIGHT
EX_EX_SYN.delay[:,:] = 'rand()*EX_EX_MAX_DELAY'

sys.stdout.write("Done\n")
sys.stdout.flush()

sys.stdout.write("Setting up intramodular excitatory-inhibitory synapses... ")
sys.stdout.flush()

EX_IN_SYN = Synapses(EX_G, IN_G,
    model='w : volt',
    on_pre='v += w',
    delay=1*ms
)

start_ex_index = 0
start_in_index = 0

for _ in range(N_MOD):
    perm = np.random.permutation(N_EX_PER_MOD)
    # i'th inhibitory neuron has input from the [i*X, (i+1)*X) excitatory
    # neurons in the same module, where X = FOCALITY, but we first permute them.
    for in_neuron in range(N_IN_PER_MOD):
        first = in_neuron*FOCALITY
        i = perm[first:first+FOCALITY]
        EX_IN_SYN.connect(i=i+start_ex_index, j=in_neuron+start_in_index)

    start_ex_index += N_EX_PER_MOD
    start_in_index += N_IN_PER_MOD

EX_IN_SYN.w[:,:] = rand() * EX_IN_SCALING * EX_IN_MAX_WEIGHT

sys.stdout.write("Done\n")
sys.stdout.flush()


sys.stdout.write("Setting up intramodular inhibitory-excitatory synapses... ")
sys.stdout.flush()

IN_EX_SYN = Synapses(IN_G, EX_G,
    model='w : volt',
    on_pre='v += w',
    delay=1*ms
)
start_ex_index = 0
start_in_index = 0
all_excitatory_neurons = np.array(range(N_EX_PER_MOD))
for _ in range(N_MOD):
    for in_neuron in range(N_IN_PER_MOD):
        IN_EX_SYN.connect(
            i=in_neuron+start_in_index, 
            j=all_excitatory_neurons+start_ex_index
        )
    start_ex_index += N_EX_PER_MOD
    start_in_index += N_IN_PER_MOD

IN_EX_SYN.w[:,:] = rand() * IN_EX_SCALING * IN_EX_MIN_WEIGHT


sys.stdout.write("Done\n")
sys.stdout.flush()

sys.stdout.write("Setting up intramodular inhibitory-inhibitory synapses... ")
sys.stdout.flush()
IN_IN_SYN = Synapses(IN_G,
    model='w : volt',
    on_pre='v += w',
    delay=1*ms
)

start_index = 0
all_inhibitory_neurons = np.array(range(N_IN_PER_MOD))
for _ in range(N_MOD):
    for in_neuron in range(N_IN_PER_MOD):
        all_inhibitory_neurons = np.array([
            n for n in range(N_IN_PER_MOD) if n != in_neuron
        ])
        IN_IN_SYN.connect(
            i=in_neuron+start_index,
            j=all_inhibitory_neurons+start_index
        )
    start_index += N_IN_PER_MOD

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


print("Running simulation..")

# Monitors
duration = 20000*ms
run(duration)


## Pickle out simulation data
#PICKLE_OUT_DATA = [
#    int(duration/ms),
#    list(M_EX.t/ms),
#    list(M_IN.t/ms),
#]
#fname = 'experiment_data/exp8conn_{}_{}-{}-{}-{}-{}.pickle'.format(
#    int(duration/ms),
#    EX_EX_INTRA_SCALING,
#    EX_EX_INTER_SCALING,
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
X1 = M_EX.t/ms
X2 = M_IN.t/ms
X = np.concatenate([X1, X2])
Y = np.concatenate([M_EX.i, M_IN.i+N_EX])

XY = [(t, s) for (t, s) in zip(X, Y) if t >= transient_thres]
XY = sorted(XY, key=lambda tup: tup[0])
X, Y = zip(*XY)

dt = 75 # ms
shift = 10 # ms
total_steps = int(duration/(shift*ms))

print('Plotting..')
plt.figure(1)
ma = psd.moving_average(X, dt, shift, total_steps)
time_scale = np.arange(0, duration/ms, shift)
time_scale = time_scale[100:-100]
plt.plot(time_scale, ma[100:-100])
plt.title('All Neurons')
plt.xlabel('Simulation Time (ms)')
plt.ylabel('Mean firing rate')

ma1 = psd.moving_average(X1, dt, shift, total_steps)
ma2 = psd.moving_average(X2, dt, shift, total_steps)

plt.figure(2)
plt.plot(time_scale, ma1[100:-100])
plt.title('Excitatory Neurons')
plt.xlabel('Simulation Time (ms)')
plt.ylabel('Mean firing rate')

plt.figure(3)
plt.plot(time_scale, ma2[100:-100])
plt.title('Inhibitory Neurons')
plt.xlabel('Simulation Time (ms)')
plt.ylabel('Mean firing rate')

plt.figure(4)
f, pxx = psd.power_spectrum(X, dt, shift, total_steps)
plt.semilogy(f, pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectrum')


plt.show()

