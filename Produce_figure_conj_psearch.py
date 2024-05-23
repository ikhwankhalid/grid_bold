"""
This script performs a parameter search of the concentration parameter and
alignment jitter under the conjunctive grid by head-direction cell hypothesis.
The results are saved and used by the file 'Produce_mainfigure_conjunctive.py'.
"""
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import settings
from utils.grid_funcs import (
    gen_offsets, gridpop_conj, traj_star
)
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from utils.utils import (
    convert_to_rhombus,
    get_hexsym
)


###############################################################################
# Parameters                                                                  #
###############################################################################
tmax_rw = settings.tmax     # duration of random walk in seconds
nn = 50                     # number of points in each direction
n_jobs = 50                 # number of jobs to run in parallel
n_trials = 4                # number of trials to average over
width0 = 5                  # minimum tuning width for conjunctive cells


###############################################################################
# Functions                                                                   #
###############################################################################
def conj_paramsearch(nn, n_trials=n_trials):
    n_kappas = int(nn*2 + 1)
    n_sigmas = int(nn + 1)
    widthmax = 60
    sigmamax = 30
    widths = np.linspace(width0, widthmax, n_kappas)
    kappas = (180 / np.pi / widths)**2
    stds = np.linspace(0, sigmamax, n_sigmas)
    gr60s_all_trials = np.zeros((n_trials, n_kappas, n_sigmas))

    for trial in range(n_trials):
        print("Trial: ", trial)

        def inner_loop(i, width, stds):
            row = np.zeros(len(stds))
            for j, std in enumerate(stds):
                ox, oy = gen_offsets(N=settings.N, kappacl=0.)
                oxr, oyr = convert_to_rhombus(ox, oy)
                trajec = traj_star(
                    settings.phbins,
                    settings.rmax,
                    settings.dt
                )
                direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
                    settings.N,
                    settings.grsc,
                    settings.phbins,
                    trajec,
                    oxr,
                    oyr,
                    propconj=settings.propconj_i,
                    kappa=(180 / np.pi / width)**2,
                    jitter=std
                )
                row[j] = get_hexsym(summed_fr, trajec)
            return row

        gr60s_this_trial = Parallel(
            n_jobs=50
        )(
            delayed(
                inner_loop
            )(i, width, stds) for i, width in enumerate(widths)
        )
        gr60s_all_trials[trial, :, :] = np.array(gr60s_this_trial)

    return gr60s_all_trials, kappas, stds


###############################################################################
# Run                                                                         #
###############################################################################
psearch_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "extend_psearch_star",
    "psearch_30sigma.pkl"
)

if not os.path.isfile(psearch_fname):
    print("Running psearch...")
    gr60s_all, kappas, stds = conj_paramsearch(nn)
    os.makedirs(os.path.dirname(psearch_fname), exist_ok=True)
    with open(psearch_fname, 'wb') as f:
        pickle.dump([gr60s_all, kappas, stds], f)
else:
    with open(psearch_fname, 'rb') as f:
        gr60s_all, kappas, stds = pickle.load(f)

###############################################################################
# Plotting                                                                    #
###############################################################################
plt.figure()
K, S = np.meshgrid(np.sqrt(1/kappas)*180/np.pi, stds)
print(gr60s_all.shape)
gr60s = np.nanmedian(gr60s_all, axis=0)
pcolor_psearch = plt.imshow(
    gr60s.T, extent=(0, 60, 0, 30), origin="lower", aspect=1,
    norm=matplotlib.colors.LogNorm()
)
gr60s = gaussian_filter(gr60s, settings.smooth_sigma)
CS = plt.contour(
    K, S, gr60s.T, levels=[10, 50, 200, 400, 600, 800, 1000], colors='white'
)
plt.gca().clabel(CS, inline=True, fontsize=16)
plt.xticks(fontsize=18.)
plt.xlim(5, 60)
plt.show()
plt.close()
