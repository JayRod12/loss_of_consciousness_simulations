import pickle
import numpy as np
import matplotlib.pyplot as plt
import power_spectral_density as psd

print("Reading pickled data..")
fname = 'experiment_data/exp8conn_20000_17-300-50-2-1.pickle'
with open(fname, 'rb') as f:
    DATA = pickle.load(f)

[duration, X1, X2] = DATA

# Discard initial transient signal
transient_thres = 1000 # ms
X = np.concatenate([X1, X2])

dt = 75 # ms
shift = 10 # ms
total_steps = int(duration/shift)

print('Plotting..')
plt.figure(1)
ma = psd.moving_average(X, dt, shift, total_steps)
plt.plot(np.arange(0, duration, shift), ma)
plt.xlabel('Mean firing rate')
plt.ylabel('Simulation Time (ms)')

plt.figure(2)
f, pxx = psd.power_spectrum(X, dt, shift, total_steps)
plt.semilogy(f, pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectrum')

plt.figure(3)
plt.plot(X, Y, '.b')
plt.xlabel("Time (ms)")
plt.ylabel("Excitatory Neuron Index")


plt.show()
