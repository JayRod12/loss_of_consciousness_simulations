import math
import numpy as np
import pickle
import scipy.io as spio

CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
N_MOD = len(XYZ)        # Number of modules

#N_NEURONS_PER_MODULE = 10

N_PER_MOD = 10          # Neurons per module
N = N_PER_MOD * N_MOD   # Total neurons

# Excitation-Inhibition percentage 80/20
EX_PROP = 0.8
N_EX_PER_MOD = int(EX_PROP * N_PER_MOD)
N_IN_PER_MOD = N_PER_MOD - N_EX_PER_MOD
N_EX = N_EX_PER_MOD * N_MOD
N_IN = N - N_EX

# 10% connectivity within a module's excitatory neurons
INTERNAL_SYNAPSE_PROP = 0.1

# Focality
FOCALITY = 4


DISTANCE = np.zeros((N_MOD,N_MOD))
for i in range(N_MOD):
    x,y,z = XYZ[i]
    for j in range(N_MOD):
        x2,y2,z2 = XYZ[j]
        d = (x-x2)**2 + (y-y2)**2 + (z-z2)**2
        DISTANCE[i][j] = math.sqrt(d)

weight = np.zeros((N_MOD, N_MOD))
delay = np.zeros((N_MOD, N_MOD))
connections = [[] for _ in range(N_EX)]

for i in range(N_MOD):
    for j in range(N_MOD):
        weight[i,j] = CIJ[i, j]
        if weight[i,j] > 0:
            connections[i].append(j)
            delay[i,j] = DISTANCE[i, j]

WRITTEN_OBJECT = [
    N, N_MOD, N_PER_MOD, N_EX, N_IN, N_EX_PER_MOD, N_IN_PER_MOD, weight, delay, connections,
    INTERNAL_SYNAPSE_PROP, FOCALITY
]

with open('precomputed_exp_8.pickle', 'wb') as f:
    pickle.dump(WRITTEN_OBJECT, f)

