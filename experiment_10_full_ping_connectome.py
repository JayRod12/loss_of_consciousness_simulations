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

#PING_CONFIG
n_ex_mod, n_in_mod = 40, 10
exin_w, inex_w, inin_w = (10,2), (10,2), (10,2)
exin_d, inex_d, inin_d = (2,1), (5,2), (5,2)

exin_conn, inex_conn, inin_conn = 0.7, 1.0, 1.0
min_d, max_d = 1, 20
min_w, max_w = 0, 15
min_sigma_w, max_sigma_w = 1, 3
min_sigma_d, max_sigma_d = 1, 3

inter_connectivity = 0.1
inter_scaling_factor = 20

def run_experiment(
        n_mod=1000,
        duration=5000,
        inter_conn=inter_connectivity,
        inter_scaling=inter_scaling_factor,
        log_scaling=False,
        save_output=False,
        verbose=False
    ):

    seed(1357)

    CIJ  = spio.loadmat('data/Conectoma.mat')['CIJ_fbden_average']
    XYZ = spio.loadmat('data/coords_sporns_2mm.mat')['coords_new']
    n_mod = min(n_mod, len(XYZ)) # Number of modules

    # Setup
    n_ex = n_ex_mod * n_mod
    n_in = n_in_mod * n_mod

    EX_G = ExcitatoryNeuronGroup(n_ex)
    IN_G = InhibitoryNeuronGroup(n_in)
#    EX_G.v = '(rand()*10 - 65) * mV'
#    IN_G.v = '(rand()*10 - 65) * mV'

    # Define all synapse objects
    echo = echo_start("Setting up synapses... \n")

    mu_w, sigma_w, mu_d, sigma_d = get_synapse_weights_and_delays(n_mod)

    # Excitatory-Inhibitory synapses within modules
    EX_IN_SYN = Synapses(EX_G, IN_G,
        model='w: volt',
        on_pre='v += w',
    )
    echo2 = echo_start('\tEX_IN_SYN... ')
    syn, weights, delays = get_synapses(
        n_mod, n_ex_mod, n_in_mod, mu_w, sigma_w, mu_d, sigma_d, exin_conn
    )
    s_i, s_j = zip(*syn)
    EX_IN_SYN.connect(i=s_i, j=s_j)
    EX_IN_SYN.w = weights
    EX_IN_SYN.delay = delays
    echo_end(echo2, "({:,} synapses)".format(len(EX_IN_SYN)))

    # Inhibitory-Excitatory synapses within modules
    IN_EX_SYN = Synapses(IN_G, EX_G,
        model='w: volt',
        on_pre='v -= w',
    )
    echo2 = echo_start('\tIN_EX_SYN... ')
    syn, weights, delays = get_synapses(
        n_mod, n_in_mod, n_ex_mod, mu_w, sigma_w, mu_d, sigma_d, inex_conn
    )
    s_i, s_j = zip(*syn)
    IN_EX_SYN.connect(i=s_i, j=s_j)
    IN_EX_SYN.w = weights
    IN_EX_SYN.delay = delays
    echo_end(echo2, "({:,} synapses)".format(len(IN_EX_SYN)))

    # Inhibitory-Inhibitory synapses within modules
    IN_IN_SYN = Synapses(IN_G,
        model='w: volt',
        on_pre='v -= w',
    )
    echo2 = echo_start('\tIN_IN_SYN... ')
    syn, weights, delays = get_synapses(
        n_mod, n_in_mod, n_in_mod, mu_w, sigma_w, mu_d, sigma_d, inin_conn
    )
    s_i, s_j = zip(*syn)
    IN_IN_SYN.connect(i=s_i, j=s_j)
    IN_IN_SYN.w = weights
    IN_IN_SYN.delay = delays

    echo_end(echo2, "({:,} synapses)".format(len(IN_IN_SYN)))


    # Inter module connections (follows connectome structure)
    # Only excitatory-excitatory connections will be created
    INTER_EX_EX_SYN = Synapses(EX_G,
        model='w: volt',
        on_pre='v += w',
    )
    echo2 = echo_start('\tINTER_EX_EX_SYN... ')

    synapses, delay_matrix = get_connectivity(n_mod, n_ex_mod, inter_conn, XYZ, CIJ)
    synapses_i, synapses_j = map(np.array, zip(*synapses))

    INTER_EX_EX_SYN.connect(i=synapses_i, j=synapses_j)

    INTER_EX_EX_SYN.delay = \
            delay_matrix[np.array(synapses_i/n_ex_mod), np.array(synapses_j/n_ex_mod)] * ms

    if log_scaling:
        INTER_EX_EX_SYN.w = np.log(CIJ[synapses_i/n_ex_mod, synapses_j/n_ex_mod]) * \
                                inter_scaling * mV
    else:
        INTER_EX_EX_SYN.w = CIJ[synapses_i/n_ex_mod, synapses_j/n_ex_mod] * \
                                inter_scaling * mV

    echo_end(echo2, "({:,} synapses)".format(len(INTER_EX_EX_SYN)))
    echo_end(echo, "All synapses created")

    echo = echo_start("Supplying Poisson input to network... ")

    # Use the same number of Poisson Inputs (40) as in PING model.
    # N = Number of poisson inputs provided to each neuron in EX_G
    POISSON_INPUT_WEIGHT=8*mV
    #PI_EX = PoissonInput(EX_G, 'v', len(EX_G), 20*Hz, weight=POISSON_INPUT_WEIGHT)
    PI_EX = PoissonInput(EX_G, 'v', 40, 20*Hz, weight=POISSON_INPUT_WEIGHT)

    echo_end(echo)

    echo = echo_start("Running sym... ")

#    recorded_neurons = [0, 1, 50, 90]
    M_EX = SpikeMonitor(EX_G)
    M_IN = SpikeMonitor(IN_G)
#    M_V = StateMonitor(EX_G, 'v', record=recorded_neurons)
    run(duration*ms)

    echo_end(echo)

    X, Y = np.array(M_EX.t/ms), np.array(M_EX.i)
    X2, Y2 = np.array(M_IN.t/ms), np.array(M_IN.i)

    DATA = {
        'X': X,
        'Y': Y,
        'X2': X2,
        'Y2': Y2,
        'duration': duration,
        'n_mod': n_mod,
        'n_ex': n_ex,
        'n_in': n_in,
        'n_ex_mod': n_ex_mod,
        'n_in_mod': n_in_mod,
        'mu_w': mu_w,
        'mu_d': mu_d,
    }
    if save_output:
        fname = "experiment_data/exp10_{}sec.pickle".format(int(duration/1000))
        echo = echo_start("Storing data to {}... ".format(fname))
        with open(fname, 'wb') as f:
            pickle.dump(DATA, f)

        echo_end(echo)
    return DATA

def get_synapse_weights_and_delays(n_mod):
    mu_w = (max_w - min_w) * np.random.random_sample((n_mod,)) + min_w
    sigma_w = (max_sigma_w - min_sigma_w) * np.random.random_sample((n_mod,)) + min_sigma_w
    mu_d = (max_d - min_d) * np.random.random_sample((n_mod,)) + min_d
    sigma_d = (max_sigma_d - min_sigma_d) * np.random.random_sample((n_mod,)) + min_sigma_d
    return mu_w, sigma_w, mu_d, sigma_d


def get_synapses(n_mod, n1, n2, mu_w, sigma_w, mu_d, sigma_d, conn):
    synapses = [
        [
            (mod_i * n1 + i,  mod_i * n2 + j)
            for i in range(n1)
            for j in range(n2)
            if sample() < conn
        ]
        for mod_i in range(n_mod)
    ]

    # Delays and weights are clipped to a minimum and maximum (because the normal
    # distribution would allow weights and delays to become negative otherwise.
    weights = np.concatenate([
        np.clip(
            np.random.normal(mu_w[mod_i], sigma_w[mod_i], size=len(synapses[mod_i])),
            min_w, max_w
        )
        for mod_i in range(n_mod)
    ]) * mV
    delays = np.concatenate([
        np.clip(
            np.random.normal(mu_d[mod_i], sigma_d[mod_i], size=len(synapses[mod_i])),
            min_d, max_d
        )
        for mod_i in range(n_mod)
    ]) * ms

    return np.concatenate(synapses), weights, delays


def get_connectivity(n_mod, n_ex_mod, inter_conn, XYZ, CIJ):
    synapses = []
    delay_matrix = np.zeros((n_mod, n_mod))
    for i in range(n_mod):
        x, y, z = XYZ[i]
        for j in range(n_mod):
            if CIJ[i][j] > 0:
                # Delay = distance / speed, speed = 2 m/s
                x2, y2, z2 = XYZ[j]
                delay_matrix[i][j] = math.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)/2
                base_i, base_j = i * n_ex_mod, j * n_ex_mod
                synapses += [(base_i + ii, base_j + jj)
                    for ii in range(n_ex_mod)
                    for jj in range(n_ex_mod)
                    if sample() < inter_conn
                ]
    return synapses, delay_matrix

def mapped_function(tup):
    n, c, w  = tup
    run_experiment(n, duration, c, w)

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--modules', help='number of modules to simulate', type=int)
    parser.add_argument('-t', '--duration', help='duration of the simulation in milliseconds', type=int)
    parser.add_argument('--inter_conn', help='inter-modular network connectivity', type=float)
    parser.add_argument('--inter_scaling', help='inter-modular weight scaling factor', type=float)
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
            inter_conn = float(args.conn)
        else:
            inter_conn = 0.1

        if args.inter_scaling:
            inter_scaling = float(args.inter_scaling)
        else:
            inter_scaling = 10

        # Data & Parameters
        save_output_to_file = False

        data = run_experiment(modules, duration, inter_conn, inter_scaling,
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


