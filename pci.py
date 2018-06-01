# Input: perturbation data 300ms/500ms
#   perturbation occurs at t_pert ms
# Output: PCI index

from brian2 import *
import pandas as pd

import power_spectral_density as psd

t_post = 300

def pci(data, dt, shift):
    X, Y, n_mod, n_ex_mod, t_pert = [
        data[k]
        for k
        in ['X', 'Y', 'n_mod', 'n_ex_mod', 't_pert']
    ]
#    t_pert = 2000

    t1, t2, t3 = 1000, t_pert, t_pert + t_post

    # Group spike by modules
    #pre_data = [[] for _ in range(n_mod)]
    #post_data = [[] for _ in range(n_mod)]

    X = pd.Series(X)
    Y = pd.Series(Y // n_ex_mod)
    gb = X.groupby(Y)

    steps1 = int((t2 - t1) / shift)
    steps2 = int((t3 - t2) / shift)
    pre_data = np.zeros(n_mod * steps1)
    post_data = np.zeros(n_mod * steps2)

    for mod in gb.groups:
        x = np.array(gb.get_group(mod))
        msk1 = x < t2
        msk2 = np.logical_and(x >= t2, x < t3)

        pre_data[mod * steps1 : (mod + 1) * steps1] = \
            psd.moving_average(x[msk1], dt, shift, t1, t2)[0]
        post_data[mod * steps2 : (mod + 1) * steps2] = \
            psd.moving_average(x[msk2], dt, shift, t2, t3)[0]
#        pre_mean[mod] = pre_data[mod].mean()
#        pre_std[mod] = pre_data[mod].std()
#        post_data[mod] = psd.moving_average(x[msk2], dt, shift, t2, t3)

    return pre_data, post_data



    # Group data by modules?

     

