from brian2 import *

def to_edge_list(adj_list):
    edge_list = []
    for i in range(len(adj_list)):
        if adj_list[i] == 1:
            edge_list.append(i)
    return edge_list

def watts_strogatz(N, k, p, eqs, thres_eq, reset_eq, method='rk4',
        on_pre_action='', delay=''):
    G = NeuronGroup(N, eqs, threshold=thres_eq, reset=reset_eq, method=method)
    S = Synapses(G, G, on_pre=on_pre_action, delay=delay)
    cij = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(i-k//2, i+k//2+1):
            j = j%N

            if i != j:
                cij[i,j] = 1
                cij[j,i] = 1
                
    # Only iterate upper triangular part to avoid modifying an entry twice
    for i in range(N):
        for j in range(i+1,N):
            if cij[i,j] == 1 and sample() < p:
                cij[i,j] = 0
                cij[j,i] = 0
                # Choose a random vertex in [i+1,N)
                h = np.random.randint(i+1,N)
                cij[i,h] = 1
                cij[h,i] = 1

    for i in range(N):
        S.connect(i=i,j=to_edge_list(cij[i]))

    return G, S, cij

