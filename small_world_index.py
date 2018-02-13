import numpy as np

MAX_INT = (1<<32)-1

# Path length averaged over all pairs of nodes    
# G adjacency matrix
def mean_path_length(G):
    L = Floyd_Warshall(G)
    N = len(G)
    total_length = 0
    for u in range(N):
        for v in range(N):
            if u == v:
                continue
            total_length += L[u,v]
    return total_length / (N*(N-1)) 

# G adjacency matrix
def Floyd_Warshall(G):
    N = len(G)
    #L = np.((N,N), (1<<32) -1, dtype=np.int32)
    L = np.copy(G)
    for i in range(N):
        for j in range(N):
            if L[i,j] == 0:
                L[i,j] = MAX_INT

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if L[i,k] + L[k,j] < L[i,j]:
                    L[i,j] = L[i,k] + L[k,j]
    return L

def clustering_coefficient_vertex(G, v):
    ns = neighbours(G,v)
    #print(v, ns)
    if len(ns) < 2:
        return len(ns)
    inter_neighbour_edges = 0
    for u in ns:
        for w in ns:
            if u != w and G[u,w]:
                inter_neighbour_edges += 1
    possible_edges = len(ns) * (len(ns) - 1) / 2 
    return float(inter_neighbour_edges) / (2*possible_edges)

# G adjacency list
def clustering_coefficient(G):
    N = len(G)
    coeff = 0
    for v in range(N):
        c = clustering_coefficient_vertex(G, v)
        #print(v, c)
        coeff += c

    return float(coeff) / N

# Get neighbours of a vertex
def neighbours(G, v):
    ns = []
    for u, is_connected in enumerate(G[v]):
        if is_connected:
            ns.append(u)
    return ns

def small_world_index(G):
    N = len(G)
    mpl = mean_path_length(G)
    cc = clustering_coefficient(G)

    k = mean_degree(G)
    rand_mpl = np.log(N)/np.log(k)
    rand_cc = float(k) / N
    
    return (float(cc)/rand_cc) / (float(mpl)/rand_mpl)

def mean_degree(G):
    N = len(G)
    degree = 0
    for v in range(N):
        degree += len(neighbours(G,v))
    return float(degree) / N

def test1():
    G = np.zeros((5,5))
    G[0,4] = 1
    G[1,2] = 1
    G[1,3] = 1
    G[1,4] = 1
    G[2,4] = 1
    G[4,0] = 1
    G[2,1] = 1
    G[3,1] = 1
    G[4,1] = 1
    G[4,2] = 1
    
    print("Mean Path Length: {}".format(mean_path_length(G)))
    print("Clustering Coefficient of G: {}".format(clustering_coefficient(G)))
    print("Small World Index of G: {}".format(small_world_index(G)))



