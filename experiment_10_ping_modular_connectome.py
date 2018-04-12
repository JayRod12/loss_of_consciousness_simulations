
from brian2 import *
from neuron_groups import *
from echo_time import *

import scipy.io as spio
import matplotlib.pyplot as plt
import power_spectral_density as psd

CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
N_MOD = len(XYZ)        # Number of modules

# Parameters
N_MOD = 200

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
    #condition='i >= ex_start and i < ex_end and j >= ex_start and j < ex_end',
    p=EX_CONNECTIVITY
)
print('\tEX_EX_SYN: {}s'.format(time.time() - tt))

tt = time.time()
EX_IN_SYN.connect(
    condition='j == int(i/4)'
#    condition='i >= ex_start and i < ex_end and j == int(i/4)'
)
print('\tEX_IN_SYN: {}s'.format(time.time() - tt))

tt = time.time()
IN_EX_SYN.connect(
    condition='int(i/10) == int(j/40)'
#    condition='i >= in_start and i < in_end and j >= ex_start and j < ex_end'
)
print('\tIN_EX_SYN: {}s'.format(time.time() - tt))

tt = time.time()
IN_IN_SYN.connect(
    condition='int(i/10) == int(j/10)',
#    condition='i >= in_start and i < in_end and j >= in_start and j < in_end',
    p=IN_CONNECTIVITY
)
print('\tIN_IN_SYN: {}s'.format(time.time() - tt))

#for module in range(N_MOD):
#    ex_start, in_start = module * N_EX_MOD, module * N_IN_MOD
#    ex_end, in_end = ex_start + N_EX_MOD, in_start + N_IN_MOD
#    #a = time.time()
#    #EX_EX_SYN.connect(
#    #    condition='i >= ex_start and i < ex_end and j >= ex_start and j < ex_end',
#    #    p=EX_CONNECTIVITY
#    #)
#    #print('EX_EX_SYN: {}s'.format(time.time() - a))
#
#    #a = time.time()
#    #EX_IN_SYN.connect(
#    #    condition='i >= ex_start and i < ex_end and j == int(i/4)'
#    #)
#    #print('EX_IN_SYN: {}s'.format(time.time() - a))
#    #a = time.time()
#    #IN_EX_SYN.connect(
#    #    condition='i >= in_start and i < in_end and j >= ex_start and j < ex_end'
#    #)
#    #print('IN_EX_SYN: {}s'.format(time.time() - a))
#    a = time.time()
#    IN_IN_SYN.connect(
#        condition='i >= in_start and i < in_end and j >= in_start and j < in_end',
#        p=IN_CONNECTIVITY
#    )
#    print('IN_IN_SYN: {}s'.format(time.time() - a))
    
EX_EX_SYN.w = EX_EX_WEIGHT
EX_IN_SYN.w = EX_IN_WEIGHT
IN_EX_SYN.w = IN_EX_WEIGHT
IN_IN_SYN.w = IN_IN_WEIGHT

echo_end(echo)


echo = echo_start("Running sym... ")
M = SpikeMonitor(EX_G)
duration = 5000*ms
run(duration)
echo_end(echo)

echo = echo_start("Post-processing and plotting... ")
MEASURE_START = 1000
MEASURE_DURATION = 500

X1, Y1 = [], []
for spike_t, spike_idx in zip(M.t/ms, M.i):
    if MEASURE_START <= spike_t < MEASURE_START + MEASURE_DURATION and \
            spike_idx < 40:
        X1.append(spike_t)
        Y1.append(spike_idx)

print("{} excitatory neuron spikes in total".format(len(X1)))
#X, Y = M.t/ms, M.i
X, Y = X1, Y1

dt = 10 # ms
shift = 5 # ms
total_steps = int(duration/(shift*ms))
ma, time_scale = psd.moving_average(X, dt, shift, total_steps, True)

X2, Y2 = [], []
for ma_val, t in zip(ma, time_scale):
    if MEASURE_START <= t[0] < MEASURE_START + MEASURE_DURATION:
        Y2.append(ma_val)
        X2.append(t[0])


plt.subplot(211)
plt.plot(X1, Y1, '.b')
plt.ylabel('Neuron Index')
plt.xlabel('Simulation Time (ms)')

plt.subplot(212)
plt.plot(X2, Y2)
plt.xlabel('Simulation Time (ms)')
plt.ylabel('Mean firing rate')

echo_end(echo)
plt.show()
    
