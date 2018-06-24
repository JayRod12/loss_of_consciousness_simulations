
"""
Calculate Perturbational Complexity Index from simulation data.
"""

Input: perturbation data 300ms/500ms
   perturbation occurs at tms_time ms
 Output: PCI index

from brian2 import *
from lz76 import LZ76
import pandas as pd

import power_spectral_density as psd

t_post = 300
t_pre = 300

def pci(data, dt, shift, method1=False):
    """
    Calculate Perturbational Complexity Index.

    Arguments:
        - data [dict]: must contain the data of a simulation, as obtained by
            calling ex.run_simulation(with_tms=True) for some experiment.
        - dt [int/float]: averaging window size for moving average calculations
        - shift [int/float]: dictates the time intervals between measurements
            in the moving average calculations.
        - method1 [bool]: If true, use Method 1, where each modules activity
            is compared against the mean network pre-stimulus rate to
            determine if a significant source of activation is present.
            Otherwise, each module's activity is compared against its own
            pre-stimulus firing rate.
    Return:
        - Normalized PCI index.
    """
    X, Y, n_mod, n_ex_mod, tms_time = [
        data[k]
        for k
        in ['X', 'Y', 'n_mod', 'n_ex_mod', 'tms_time']
    ]

    t1, t2, t3 = tms_time - t_pre, tms_time, tms_time + t_post

    # Group spike by modules
    X = pd.Series(X)
    Y = pd.Series(Y // n_ex_mod)
    gb = X.groupby(Y)

    steps1 = int((t2 - t1) / shift)
    steps2 = int((t3 - t2) / shift)
    pre_data = np.zeros(n_mod * steps1)
    post_data = np.zeros(n_mod * steps2)

    for mod in gb.groups:
        # Get spikes associated to module ``mod``
        x = np.array(gb.get_group(mod))
        msk1 = x < t2
        msk2 = np.logical_and(x >= t2, x < t3)

        # Pre and post TMS moving average for the given module
        pre_ma = psd.moving_average(x[msk1], dt, shift, t1, t2)[0]
        post_ma = psd.moving_average(x[msk2], dt, shift, t2, t3)[0]
        if method1:
            pre_data[mod * steps1 : (mod + 1) * steps1] = pre_ma
            post_data[mod * steps2 : (mod + 1) * steps2] = post_ma
        else:
            params = pre_ma.mean() + 2 * pre_ma.std()
            post_data[mod * steps2 : (mod + 1) * steps2] = (post_ma > params).astype(int)

    # PCI
    if method1:
        pre_mean, pre_std = pre_data.mean(), pre_data.std()
        post_binarized = (post_data > pre_mean + 2 * pre_std).astype(int)
        lz_complexity = LZ76(post_binarized) * np.log(len(post_data)) / len(post_data)
    else:
        post_binarized = post_data
        lz_complexity = LZ76(post_binarized) * np.log(len(post_data)) / len(post_data)

    return lz_complexity

