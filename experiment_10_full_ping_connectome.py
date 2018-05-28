
from brian2 import *
from lz76 import LZ76
from tqdm import tqdm
from echo_time import *
from neuron_groups import *
from operator import itemgetter
from multiprocessing import Pool

import pickle
import argparse
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import power_spectral_density as psd


def run_experiment(n_mod, duration, connectivity, scaling_factor, log_scaling=False, save_output=False, verbose=False):
    seed(1357)

    CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
    XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
#    N_MOD = len(XYZ)        # Number of modules
    N_MOD = min(n_mod, len(XYZ)) # Number of modules

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

    if verbose:
        echo = echo_start("Setting up synapses... \n")

    tt = time.time()
    EX_EX_SYN.connect(
        condition='int(i/40) == int(j/40)',
        p=EX_CONNECTIVITY
    )
    EX_EX_SYN.w = EX_EX_WEIGHT
    if verbose:
        print('\tEX_EX_SYN ({:,} synapses): {}s'.format(len(EX_EX_SYN.w), time.time() - tt))

    tt = time.time()
    EX_IN_SYN.connect(
        condition='j == int(i/4)'
    )
    EX_IN_SYN.w = EX_IN_WEIGHT
    if verbose:
        print('\tEX_IN_SYN ({:,} synapses): {}s'.format(len(EX_IN_SYN.w), time.time() - tt))

    tt = time.time()
    IN_EX_SYN.connect(
        condition='int(i/10) == int(j/40)'
    )
    IN_EX_SYN.w = IN_EX_WEIGHT
    if verbose:
        print('\tIN_EX_SYN ({:,} synapses): {}s'.format(len(IN_EX_SYN.w), time.time() - tt))

    tt = time.time()
    IN_IN_SYN.connect(
        condition='int(i/10) == int(j/10)',
        p=IN_CONNECTIVITY
    )
    IN_IN_SYN.w = IN_IN_WEIGHT
    if verbose:
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
                    if sample() < connectivity
                ]

    synapses_i, synapses_j = map(np.array, zip(*synapses))
    INTER_EX_EX_SYN.connect(i=synapses_i, j=synapses_j)

    INTER_EX_EX_SYN.delay = \
            delay_matrix[np.array(synapses_i/N_EX_MOD), np.array(synapses_j/N_EX_MOD)] * ms

    if log_scaling:
        INTER_EX_EX_SYN.w = np.log(CIJ[synapses_i/N_EX_MOD, synapses_j/N_EX_MOD]) * \
                                scaling_factor * mV
    else:
        INTER_EX_EX_SYN.w = CIJ[synapses_i/N_EX_MOD, synapses_j/N_EX_MOD] * \
                                scaling_factor * mV

    if verbose:
        print('\tINTER_EX_EX_SYN ({:,} synapses): {}s'.format(len(INTER_EX_EX_SYN.w), time.time() - tt))
        echo_end(echo)
        echo = echo_start("Running sym... ")

    recorded_neurons = [0, 1, 50, 90]
    M_EX = SpikeMonitor(EX_G)
    M_IN = SpikeMonitor(IN_G)
    M_V = StateMonitor(EX_G, 'v', record=recorded_neurons)
    run(duration*ms)

    if verbose:
        echo_end(echo)
        echo = echo_start("Processing spike data...")

    # Problem: inhibitory indexes overlap with excitatory indexes
    #X = np.concatenate(M_EX.t/ms, M_IN.t/ms)
    #Y = np.concatenate(M_EX.i, M_IN.i)
    #perm_sort = X.argsort()
    #X, Y = X[perm_sort], Y[perm_sort]
    X, Y = M_EX.t/ms, M_EX.i

    start_time = 1000
    end_time = duration

    if verbose:
        echo_end(echo)
        echo = echo_start("Selecting spikes between {}ms and {}ms of simulation... "
            .format(start_time, end_time))

    # Discard transient
    mask = np.logical_and(X >= 1000, X < end_time)
    X, Y = X[mask], Y[mask]

    if verbose:
        echo_end(echo)
        echo = echo_start("Separating list of spikes into separate lists for each module... ")


    #modules = [[] for _ in range(N_MOD)]
    #for spike_t, spike_idx in zip(X, Y):
    #    modules[spike_idx // N_EX_MOD].append(spike_t)

    X_series, Y_series = pd.Series(X), pd.Series(Y // N_EX_MOD)
    gb = X_series.groupby(Y_series)
    modules = [[] for _ in range(N_MOD)]
    for mod in gb.groups:
        modules[mod] = np.array(gb.get_group(mod))

    if verbose:
        echo_end(echo)
        echo = echo_start("Calculating Lempel Ziv Complexity of firing rates... ")

    dt = 75 # ms
    shift = 10 # ms

    lz_comp = np.zeros(N_MOD)
    for mod in range(N_MOD):
        x, _ = psd.moving_average(modules[mod], dt, shift, start_time, end_time)
        binx = (x > x.mean()).astype(int)
        lz_comp[mod] = LZ76(binx)

    if verbose:
        echo_end(echo)

    n_steps = float(end_time - start_time) / shift

    params = (n_mod, duration // 1000, connectivity, scaling_factor, int(log_scaling))
    plt.figure()
    plt.hist(lz_comp*np.log(n_steps)/n_steps)
    plt.xlabel('Normalized LZ complexity')
    plt.ylabel('Module counts')
    plt.savefig('figures/sim10/sim_10_sweep_lz_complexity_N{}_{}s_{}_{}_{}.png'.format(*params))

    plt.figure()
    mask = np.logical_and.reduce((X >= 3000, X < 3500, Y < 150))
    plt.plot(X[mask], Y[mask], '.')
    plt.xlabel('Simulation Time (ms)')
    plt.savefig('figures/sim10/sim_10_sweep_raster_N{}_{}s_{}_{}_{}.png'.format(*params))

    plt.figure()
    t = M_V.t/ms
    mask = np.logical_and(t >= 1000, t < 1100)
    t = t[mask]
    for idx, neuron_idx in enumerate(recorded_neurons):
        v = M_V.v[idx]
        v = v[mask]

        plt.subplot(2,2,idx+1)
        plt.plot(t, v)
        plt.ylabel('Neuron {} voltage'.format(neuron_idx))
        plt.xlabel('Simulation time (ms)')

    plt.tight_layout()
    plt.savefig('figures/sim10/sim_10_sweep_neuron_voltage_N{}_{}s_{}_{}_{}.png'.format(*params))

    DATA = {
        'X': X,
        'Y': Y,
        'duration': duration,
        'N_MOD': N_MOD,
        'M_V': M_V,
        'M_EX': M_EX,
        'M_IN': M_IN,
        'lz_comp': lz_comp,
        'n_steps': n_steps,
    }
    if save_output:
        fname = "experiment_data/exp10_{}sec.pickle".format(int(duration/1000))
        if verbose:
            echo = echo_start("Storing data to {}... ".format(fname))
        with open(fname, 'wb') as f:
            pickle.dump(DATA, f)

        if verbose:
            echo_end(echo)
    return DATA


def mapped_function(tup):
    N, c, w  = tup
    run_experiment(N, duration, c, w)

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--modules', help='number of modules to simulate', type=int)
    parser.add_argument('-t', '--duration', help='duration of the simulation in milliseconds', type=int)
    parser.add_argument('-c', '--conn', help='inter-modular network connectivity', type=float)
    parser.add_argument('-w', '--scaling_factor', help='inter-modular weight scaling factor', type=float)
    parser.add_argument('--log_scaling', help='use logarithmic inter-modular weight scaling', action='store_true')
    parser.add_argument('-v', '--verbose', help='use logarithmic inter-modular weight scaling', action='store_true')
    parser.add_argument('--sweep', help='Do a parameter sweep, ignores all other arguments', action='store_true')
    args = parser.parse_args()

    if not args.sweep:
        if args.modules:
            modules = args.modules
        else:
            modules = 998

        if args.duration:
            duration = float(args.duration)
        else:
            duration = 5000 # ms

        if args.conn:
            connectivity = float(args.conn)
        else:
            connectivity = 0.1

        if args.scaling_factor:
            scaling_factor = float(args.scaling_factor)
        else:
            scaling_factor = 10

        # Data & Parameters
        save_output_to_file = False

        data = run_experiment(modules, duration, connectivity, scaling_factor,
                args.log_scaling, save_output_to_file, args.verbose)
    else:
        duration = 5000
        p = Pool(5)

        params = [
            (N, c, w)
            for N in np.logspace(2, 3, 2).astype(int)
            for w in np.logspace(0, 2, 6)
            for c in np.linspace(0.01, 0.1, 2)
        ]
        p.map(mapped_function, params)


