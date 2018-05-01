
from brian2.units import *
from echo_time import *
from lz76 import LZ76
from tqdm import tqdm

import pickle
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import power_spectral_density as psd


fname = "experiment_data/exp10_20sec.pickle"
echo = echo_start("Reading data from {}... ".format(fname))

try:
    with open(fname, 'rb') as f:
        DATA = pickle.load(f)
except Exception as e:
    print('Error', e)
    exit()

echo_end(echo)

X = DATA['X']
Y = DATA['Y']
duration = DATA['duration']
N_MOD = DATA['N_MOD']
N_EX_MOD = 40
N_IN_MOD = 10

# Remove first second of simulation (for data cleaning)
start_time = 1000
end_time = 20000

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
plt.show()



