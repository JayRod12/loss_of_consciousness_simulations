
from brian2.units import *
from echo_time import *

import pickle
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
duration = DATA['duration']/ms

echo = echo_start("Post-processing and plotting... \n")
MEASURE_START = 1000
MEASURE_DURATION = 500

print("\t{:,} excitatory neuron spikes in total".format(len(X)))

tt = time.time()
X1, Y1 = [], []
for spike_t, spike_idx in zip(X, Y):
    if MEASURE_START <= spike_t < MEASURE_START + MEASURE_DURATION and \
            spike_idx < 3*40:
        X1.append(spike_t)
        Y1.append(spike_idx)

print('\tCollect relevant spikes: {}s'.format(time.time() - tt))

#X, Y = M.t/ms, M.i
X, Y = X1, Y1

tt = time.time()
dt = 10 # ms
shift = 5 # ms
total_steps = int(duration/(shift*ms))
ma, time_scale = psd.moving_average(X, dt, shift, total_steps, True)
print('\tCalculate moving average: {}s'.format(time.time() - tt))

tt = time.time()
X2, Y2 = [], []
for ma_val, t in zip(ma, time_scale):
    if MEASURE_START <= t[0] < MEASURE_START + MEASURE_DURATION:
        Y2.append(ma_val)
        X2.append(t[0])
print('\tCollect relvant moving average data points: {}s'.format(time.time() - tt))


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
    
