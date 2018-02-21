
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_constants import *
from brian2 import *
from visualise import visualise_connectivity
import power_spectral_density as psd

# Use Inhibitory equations for the inhibitory neurons as well as negative weights

EX_EX_WEIGHT = 1*mV
IN_EX_MIN_WEIGHT = 1*mV
EX_IN_MAX_WEIGHT = 1*mV
IN_IN_MIN_WEIGHT = -1*mV

EX_EX_SCALING = 17
EX_IN_SCALING = 50
IN_EX_SCALING = 2
IN_IN_SCALING = 1

EX_EX_MAX_DELAY = 20*ms

N_MOD = 8
N_MOD_CONN = 1000 # within each module of 100 neurons
N_IN = 200
N_EX = 800

# Number of excitatory neurons connecting to a single inhibitory neuron
FOCALITY = 4

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
seed(1793)
 
EX_EX_SYN = Synapses(EX_G, on_pre='v += EX_EX_SCALING * EX_EX_WEIGHT')
for mod in range(N_MOD):
    N = N_EX//N_MOD
    i = np.random.randint(N, size=N_MOD_CONN) + N * mod
    j = np.random.randint(N, size=N_MOD_CONN) + N * mod
    EX_EX_SYN.connect(i=i, j=j)

# Set delay for above created connections
EX_EX_SYN.delay[:,:] = 'rand()*EX_EX_MAX_DELAY'

# Focal excitatory-inhibitory connections
# 4 to 1 ex-in conn
EX_IN_SYN = Synapses(EX_G, IN_G,
    on_pre='v += rand() * (EX_IN_SCALING * EX_IN_MAX_WEIGHT)',
    delay=1*ms
)
perm = np.random.permutation(N_IN)
for ex_group, in_neuron in enumerate(perm):
    i = [ex_group*FOCALITY+k for k in range(FOCALITY)]
    #i = [in_neuron*FOCALITY+k for k in range(FOCALITY)]
    # Inhibitory neurons start after N_EX neurons
    j=in_neuron
    EX_IN_SYN.connect(i=i, j=j)


# Diffuse inhibitory-excitatory connections
IN_EX_SYN = Synapses(IN_G, EX_G,
    on_pre='v -= (IN_EX_SCALING * rand() * IN_EX_MIN_WEIGHT)',
    delay=1*ms
)
for in_neuron in range(N_IN):
    IN_EX_SYN.connect(i=in_neuron, j=range(N_EX))

# Diffuse inhibitory-inhibitory connections
IN_IN_SYN = Synapses(IN_G,
    on_pre='v += (IN_IN_SCALING * rand() * IN_IN_MIN_WEIGHT)',
    delay=1*ms
)

for in_neuron in range(N_IN):
    IN_IN_SYN.connect(i=in_neuron, j=np.arange(N_IN))


POISSON_INPUT_WEIGHT=2*mV
PI_EX = PoissonInput(EX_G, 'v', len(EX_G), 1*Hz, weight=POISSON_INPUT_WEIGHT)
PI_IN = PoissonInput(IN_G, 'v', len(IN_G), 1*Hz, weight=POISSON_INPUT_WEIGHT)
EX_G.v = -65*mV
IN_G.v = -65*mV

M_EX = SpikeMonitor(EX_G)
M_IN = SpikeMonitor(IN_G)

# Monitors
duration = 1000*ms
run(duration)

# Moving average
spikes = M_EX.i
times = M_EX.t/ms

# Separate spikes into their modules
spikes_per_module = [[] for _ in range(N_MOD)]
for idx, t in enumerate(times):
    mod = spikes[idx] // (N_EX / N_MOD)
    spikes_per_module[mod].append(t)

dt = 75 # ms
shift = 10 # ms
ts = np.arange(0, duration/ms, shift)

ax1 = plt.subplot(313)
for i, ss in enumerate(spikes_per_module):
    ma = psd.moving_average(ss, dt, shift, int((duration/ms)/shift))
    plt.plot(ts, ma, color="C{}".format(i), label="Module {}".format(i))
ax1.set_xlim(0, duration/ms)
ax1.set_ylabel("Mean firing rate")
ax1.set_xlabel("Time (ms)")


ax2 = plt.subplot(311)
plt.plot(M_EX.t/ms, M_EX.i, '.b') 
ax2.set_xlim(0, duration/ms)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Excitatory Neuron Index")

ax3 = plt.subplot(312)
plt.plot(M_IN.t/ms, M_IN.i, '.k') 
ax3.set_xlim(0, duration/ms)
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Inhibitory Neuron Index")

ax1.legend()


plt.figure(2)
ax1 = plt.subplot(211)
f, pxx = psd.power_spectrum(times, dt, shift, int((duration/ms)/shift))
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power Spectrum')
plt.plot(f, pxx, label="All excitatory spikes")
ax1.legend()

ax2 = plt.subplot(212)
for i, ss in enumerate(spikes_per_module):
    f, pxx = psd.power_spectrum(ss, dt, shift, int((duration/ms)/shift))
    plt.plot(f, pxx, color="C{}".format(i), label="Module {}".format(i))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power Spectrum')
ax2.legend()
plt.tight_layout()

plt.show()


