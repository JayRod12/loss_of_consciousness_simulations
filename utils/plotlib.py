"""
Utility containing different plotting functions.
"""

from lz76 import LZ76
from echo_time import *
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import power_spectral_density as psd


plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('lines', linewidth=3)
plt.rc('legend', fontsize=14)


def _calc_lz_mod(mod, modules, dt, shift, start, end):
    """
        Calculates the Lempel-Ziv complexity of a single module.
    """
    x, _ = psd.moving_average(modules[mod], dt, shift, start, end)
    binx = (x > x.mean()).astype(int)
    return LZ76(binx)

def get_lz_comp(data, start, end, dt, shift):
    """
        Calculates the Lempel-Ziv complexity of all the modules in the
        simulation, using multiple cores.
    """
    n_mod, n_ex_mod, X, Y = [
        data[k]
        for k
        in ['n_mod', 'n_ex_mod', 'X', 'Y']
    ]

    e = echo_start("Separating spikes into modules... ")
    X_series, Y_series = pd.Series(X), pd.Series(Y // n_ex_mod)
    gb = X_series.groupby(Y_series)
    modules = [[] for _ in range(n_mod)]
    for mod in gb.groups:
        modules[mod] = np.array(gb.get_group(mod))
    echo_end(e)
    
    
    f = partial(_calc_lz_mod,
        modules=modules,
        dt=dt,
        shift=shift,
        start=start,
        end=end
    )

    e = echo_start("Calculating LZ complexity of modules... ")
    pool = Pool(6)
    lz_comp = pool.map(f, np.arange(n_mod))
    pool.close()
    echo_end(e)

    return np.array(lz_comp)

def plot_lz(lz_comp, start, end, dt, shift, ax=None, **kwargs):
    """
        Plot Lempel-Ziv complexity.
        Takes as input LZ complexity as obtained from calling the method
        ``get_lz_comp`` above.
    """
    n_steps = float(end - start) / shift
    if not ax:
        _, ax = plt.subplots() 
    #y, binedges = np.histogram(lz_comp, bins=100) 
    #bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    #ax.plot(bincenters, y*np.log(n_steps)/n_steps, '.', label=label)
    kwargs.update(dict(
        #histtype='stepfilled',
        alpha=0.75,
        normed=True,
        bins=20,
    ))
    ax.hist(lz_comp*np.log(n_steps)/n_steps, **kwargs)

#    ax.hist(lz_comp*np.log(n_steps)/n_steps, label=label, bins=100)


def plot_ma(n, x, dt, shift, ax=None, start=None, end=None,**kwargs):#color=None, label=None):
    """
        Plots the moving average firing rate of a simulation, given the times associated
        to spikes as ``x``. The output is normalized as a percentage of the number of neurons
        ``n``. See psd.moving_average for details of the use of ``dt`` and ``shift``.
    """
    if not start:
        start = 1000
    if not end:
        end = max(start, max(x))
    ma, t = psd.moving_average(x, dt, shift, start, end)
    ma = 100.0 * ma / n
    if not ax:
        _, ax = plt.subplots()
        
    #ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('Firing rate\n (% neurons/ms)')
    ax.plot(t, ma, **kwargs)#color=color, label=label)
    
def plot_spectrum(x, dt, shift, ax=None, color=None, label=None):
    """ 
        Plots the power-frequency spectrum of a simulation in a semilog plot.
    """
    #dt, shift = 75, 10
    if not ax:
        _, ax = plt.subplots()
        
    start, end = 1000, max(x)
    f, pxx = psd.power_spectrum(x, dt, shift, start, end)
    ax.semilogy(f, pxx, color=color, label=label)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')


def plot_sim(data, start=1000, end=2000, min_mod=None, max_mod=None, save=None):
    """
        Summary plot of a simulation showing three plots:
            - Top: Raster plot of spikes
            - Middle: Moving average firing rate
            - Bottom: Power-frequency spectrum
        The modules that are shown in the top two plots can be controlled via ``min_mod``
        and ``max_mod``, the minimum numbered and maximum numbered modules. ``start``
        and ``end`` control the simulation time plotted.
    """
    n_ex, n_in, X, Y, X2, Y2 = [
        data[k]
        for k
        in ['n_ex', 'n_in', 'X', 'Y', 'X2', 'Y2']
    ]

    n_mod = data.get('n_mod', 1)
    n_ex_mod = data.get('n_ex_mod', n_ex)
    n_in_mod = data.get('n_in_mod', n_in)

    max_ex_y = 0
    max_in_y = 0

    if min_mod:
        min_ex_y = n_ex_mod * min(n_mod, min_mod)
        min_in_y = n_in_mod * min(n_mod, min_mod)
    else:
        min_ex_y = 0
        min_in_y = 0

    if max_mod:
        max_ex_y = n_ex_mod * min(n_mod, max_mod)
        max_in_y = n_in_mod * min(n_mod, max_mod)
    else:
        max_ex_y = n_ex_mod * n_mod
        max_in_y = n_in_mod * n_mod

    n_ex = max_ex_y - min_ex_y
    n_in = max_in_y - min_in_y

    start_time = start
    end_time = end
    
    print("{:,} exc spikes, {:,} inh spikes".format(len(X), len(X2)))
    
    mask = np.logical_and.reduce((X >= start_time, X < end_time, Y >= min_ex_y, Y < max_ex_y))
    mask2 = np.logical_and.reduce((X2 >= start_time, X2 < end_time, Y2 >= min_in_y, Y2 < max_in_y))
    
    fig, axarr = plt.subplots(3, figsize=(15,15))
    axarr[0].plot(X[mask], Y[mask]-min_ex_y, '.', color='C0', label='Excitatory Population')
    axarr[0].plot(X2[mask2], Y2[mask2]-min_in_y+(max_ex_y-min_ex_y), '.', color='C1', label='Inhibitory Population')
    axarr[0].set_ylabel('Neuron index')
    axarr[0].set_xlabel('Simulation Time (ms)')
    axarr[0].set_title('Raster plot of spikes')
    axarr[0].set_xlim([start_time, end_time])
    
    dt, shift = 5, 5
    plot_ma(n_ex, X[mask], dt, shift, ax=axarr[1] , label='Excitatory Population')
    plot_ma(n_in, X2[mask2], dt, shift, ax=axarr[1], label='Inhibitory Population')
    axarr[1].set_xlabel('Simulation Time (ms)')
    axarr[1].set_xlim([start_time, end_time])
    axarr[1].legend()
    
    plot_spectrum(X, dt, shift, ax=axarr[2], label='Excitatory Population')
    plot_spectrum(X2, dt, shift, ax=axarr[2], label='Inhibitory Population')
    axarr[2].set_xticks(np.arange(0, (1000.0/shift)/2.0 + 1, 10))
    axarr[2].legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def plot_modules(data, module_list, dt, shift, start=1000, end=2000, ax = None):
    """
        Method to plot a subset of the modules from the simulation in terms of their
        moving average firing rates.
    """
    n_ex, n_in, n_mod, n_ex_mod, n_in_mod, X, Y, X2, Y2 = [
        data[k]
        for k
        in ['n_ex', 'n_in', 'n_mod', 'n_ex_mod', 'n_in_mod', 'X', 'Y', 'X2', 'Y2']
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 6))

    dt, shift = 50, 10

    X_series, Y_series = pd.Series(X), pd.Series(Y // n_ex_mod)
    gb = X_series.groupby(Y_series)
    for i, mod in enumerate(module_list):
        plot_ma(40, np.array(gb.get_group(mod)), dt, shift, ax=ax,start=start, end=end, label='Module {}'.format(mod))
#        ma, t = psd.moving_average(np.array(gb.get_group(mod)), dt, shift, start, end)

#        ax.plot(t, ma, label='Module {}'.format(mod))
#    ax.legend()


def show_sim_stats(data):
    """
        Print a few statistics about a simulation.
    """
    X, X2 = data['X'], data['X2']
    n1, n2 = data['n_ex'], data['n_in']
    duration = data['duration']
    dt, shift = 5, 5
    start = 1000
    end = duration
    ma, t = psd.moving_average(X, dt, shift, start, end)
    ma2, _ = psd.moving_average(X2, dt, shift, start, end)

    mean1 = 100.0 * ma.mean() / n1
    mean2 = 100.0 * ma2.mean() / n2
    mean_tot = 100.0 * (ma + ma2).mean() / (n1+n2)
    print("Simulation lasted {} seconds".format(duration//1000))
    print("Number of spikes")
    print("\tExcitatory: {:,} spikes".format(len(X)))
    print("\tInhibitory: {:,} spikes".format(len(X2)))
    print("\tTotal: {:,} spikes".format(len(X2) + len(X)))
    print("Mean firing rate of the network:")
    print("\tExcitatory population: {:.2f}%".format(mean1))
    print("\tInhibitory population: {:.2f}%".format(mean2))
    print("\tAll neurons: {:.2f}%".format(mean_tot))
    
    

