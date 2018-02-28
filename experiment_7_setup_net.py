import math
import numpy as np
import pickle
import scipy.io as spio

CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
N = len(XYZ)

# Excitation-Inhibition percentage 80/20
EX_PROP = 0.8
is_excitatory = np.array([np.random.random_sample() < EX_PROP for _ in range(N)])
EX_NEURONS = np.arange(N)[is_excitatory]
IN_NEURONS = np.arange(N)[~is_excitatory]

N_EX = EX_NEURONS.size
N_IN = N - N_EX 

DISTANCE = np.zeros((N,N))
for i in range(N):
    x,y,z = XYZ[i]
    for j in range(N):
        x2,y2,z2 = XYZ[j]
        d = (x-x2)**2 + (y-y2)**2 + (z-z2)**2
        DISTANCE[i][j] = math.sqrt(d)

weight = np.zeros((N_EX, N_EX))
delay = np.zeros((N_EX, N_EX))
connections = [[] for _ in range(N_EX)]

for i in range(N_EX):
    for j in range(N_EX):
        weight[i,j] = CIJ[EX_NEURONS[i], EX_NEURONS[j]]#*EX_EX_SCALING
        if weight[i,j] > 0:
            connections[i].append(j)
            delay[i,j] = DISTANCE[EX_NEURONS[i],EX_NEURONS[j]]#/EX_EX_SCALING_DELAY

WRITTEN_OBJECT = [
    N, N_EX, N_IN, EX_NEURONS, IN_NEURONS, weight, delay, connections
]

with open('precomputed_exp_7.pickle', 'wb') as f:
    pickle.dump(WRITTEN_OBJECT, f)

