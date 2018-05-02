
from brian2 import *
from neuron_groups import *
from echo_time import *

import pickle
import argparse
import scipy.io as spio
import matplotlib.pyplot as plt
import power_spectral_density as psd

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--conn', help='inter-modular network connectivity', type=float)
parser.add_argument('--scaling_factor', help='inter-modular weight scaling factor', type=float)
parser.add_argument('--log_scaling', help='use logarithmic inter-modular weight scaling', action='store_true')
args = parser.parse_args()


# Parameters to regulate
INTER_MODULE_CONNECTIVITY = 0.1
if args.conn:
    INTER_MODULE_CONNECTIVITY = float(args.conn)

LOG_SCALING = args.log_scaling

SCALING_FACTOR_INTER_EX_EX = 10
if args.scaling_factor:
    SCALING_FACTOR_INTER_EX_EX = float(args.scaling_factor)


# Data & Parameters
SAVE_OUTPUT_TO_FILE = False

CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
N_MOD = len(XYZ)        # Number of modules

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

INTER_EX_EX_SYN.w = CIJ[synapses_i/N_EX_MOD, synapses_j/N_EX_MOD] * \
                        SCALING_FACTOR_INTER_EX_EX * mV

print('\tINTER_EX_EX_SYN ({:,} synapses): {}s'.format(len(INTER_EX_EX_SYN.w), time.time() - tt))

echo_end(echo)

echo = echo_start("Running sym... ")
M = SpikeMonitor(EX_G)
duration = 20000*ms
run(duration)
echo_end(echo)

# Complexity measures
X = M.t/ms
Y = M.i
duration = duration/ms
start_time = 1000
end_time = 20000

if SAVE_OUTPUT_TO_FILE:
    fname = "experiment_data/exp10_{}sec.pickle".format(int(duration/ms/1000))
    echo = echo_start("Storing data to {}... ".format(fname))
    DATA = {
        'X': X,
        'Y': Y,
        'duration': duration,
        'N_MOD': N_MOD
    }
    with open(fname, 'wb') as f:
        pickle.dump(DATA, f)

    echo_end(echo)


echo = echo_start("Removing first 1000ms of simulation... ")
index = 0
while X[index] < start_time:
    index += 1
X = X[index:]
Y = Y[index:]
echo_end(echo)

echo = echo_start("Separating list of spikes into separate lists for each module... ")
modules = [[] for _ in range(N_MOD)]
for spike_t, spike_idx in zip(X, Y):
    modules[spike_idx // N_EX_MOD].append(spike_t)
echo_end(echo)

dt = 75 # ms
shift = 10 # ms

echo = echo_start("Calculating Lempel Ziv Complexity of firing rates... ")

lz_comp = np.zeros(N_MOD)
for mod in tqdm(range(N_MOD)):
    x, _ = psd.moving_average(modules[mod], dt, shift, start_time, end_time)
    binx = (x > x.mean()).astype(int)
    lz_comp[mod] = LZ76(binx)

echo_end(echo)


n_steps = float(end_time - start_time) / shift
plt.hist(lz_comp*np.log(n_steps)/n_steps)
plt.xlabel('Normalized LZ complexity')
plt.ylabel('Module counts')
plt.savefig('lz_complexity_{}s_{}_{}.png'.format(int(duration/1000), INTER_MODULE_CONNECTIVITY, SCALING_FACTOR_INTER_EX_EX))
