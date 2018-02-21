
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_constants import *
from brian2 import *
from visualise import visualise_connectivity


EX_EX_WEIGHT = 1*mV
IN_EX_MIN_WEIGHT = -1*mV
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

G = NeuronGroup(N_EX+N_IN,
    EXCITATORY_NEURON_EQS,
    threshold=THRES_EQ,
    reset=EXCITATORY_RESET_EQ,
    method='rk4'
)
EX_EX_SYN = Synapses(G, on_pre='v += EX_EX_SCALING * EX_EX_WEIGHT')
for mod in range(N_MOD):
    N = N_EX//N_MOD
    i = np.random.randint(N, size=N_MOD_CONN) + N * mod
    j = np.random.randint(N, size=N_MOD_CONN) + N * mod
    EX_EX_SYN.connect(i=i, j=j)

# Set delay for above created connections
EX_EX_SYN.delay[:,:] = 'rand()*EX_EX_MAX_DELAY'

# Focal excitatory-inhibitory connections
# 4 to 1 ex-in conn
EX_IN_SYN = Synapses(G,
    on_pre='v += EX_IN_SCALING * rand() * EX_IN_MAX_WEIGHT',
    delay=1*ms
)
perm = np.random.permutation(N_IN)
for in_neuron in perm:
    i = [in_neuron*FOCALITY+k for k in range(FOCALITY)]
    # Inhibitory neurons start after N_EX neurons
    j=N_EX+in_neuron
    EX_IN_SYN.connect(i=i, j=j)


# Diffuse inhibitory-excitatory connections
IN_EX_SYN = Synapses(G,
    on_pre='v += IN_EX_SCALING * rand() * IN_EX_MIN_WEIGHT',
    delay=1*ms
)
for in_neuron in range(N_IN):
    IN_EX_SYN.connect(i=N_EX+in_neuron, j=range(N_EX))


# Diffuse inhibitory-inhibitory connections
IN_IN_SYN = Synapses(G,
    on_pre = 'v += IN_IN_SCALING * rand() * IN_IN_MIN_WEIGHT',
    delay=1*ms
)

for in_neuron in range(N_IN):
    IN_IN_SYN.connect(i=N_EX+in_neuron, j=np.arange(N_IN)+N_EX)


POISSON_INPUT_WEIGHT=2*mV
PI_EX = PoissonInput(G, 'v', len(G), 1*Hz, weight=POISSON_INPUT_WEIGHT)
G.v = -65*mV

M = SpikeMonitor(G)

# Monitors
duration = 1000*ms
run(duration)

plt.plot(M.t/ms, M.i, '.b') 
plt.show()
    


