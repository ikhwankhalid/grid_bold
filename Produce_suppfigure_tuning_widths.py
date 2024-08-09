"""
This script compiles Figure 2-figure supplement of the manuscript.
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
import settings
import matplotlib as mpl
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from utils.grid_funcs import (
    convert_to_rhombus,
    gen_offsets,
    gridpop_conj,
    traj
)
from utils.utils import ax_pos
mpl.rcParams['figure.dpi'] = 200


###############################################################################
# Parameters                                                                  #
###############################################################################
rep = 10
n_jobs = 5
n_points = 200
phbins = int(360 / 6)
wide_ang = 55.
narrow_ang = 8.1
real_ang = 28.6
savename = os.path.join(
    settings.loc,
    "conjunctive",
    "tuning_widths",
    "tuning_widths.npy"
)
psearch_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "tuning_widths",
    "psearch.pkl"
)
N = 1
kappacs = np.logspace(-2, 4, n_points)


###############################################################################
# Functions                                                                   #
###############################################################################
def get_ang_stddev(angles, strengths):
    # convert signals and angles into vectors
    vectors_x = strengths * np.cos(angles)
    vectors_y = strengths * np.sin(angles)

    # calculate the resultant vector
    resultant_x = np.sum(vectors_x)
    resultant_y = np.sum(vectors_y)

    length = np.sqrt(resultant_x**2 + resultant_y**2) / np.sum(
        np.sqrt(vectors_x**2 + vectors_y**2)
    )

    # calculate the angular standard deviation
    angular_stddev = np.sqrt(
        -2 * np.log(length)
    )

    return angular_stddev


def mfunc(i):
    ang_stddevs = np.zeros(n_points)

    # random offsets
    ox_rand, oy_rand = gen_offsets(N=N, kappacl=0.)
    oxr_rand, oyr_rand = convert_to_rhombus(ox_rand, oy_rand)

    for j, kappa in enumerate(kappacs):
        trajec = traj(settings.dt, settings.tmax)
        fr_mean, summed_fr = gridpop_conj(
            N,
            settings.grsc,
            phbins,
            trajec,
            oxr_rand,
            oyr_rand,
            propconj=settings.propconj_i,
            kappa=kappa,
            jitter=settings.jitterc_i
        )[1::2]
        angles = np.linspace(-np.pi, np.pi, phbins)
        ang_stddevs[j] = get_ang_stddev(angles, fr_mean)

    return ang_stddevs


###############################################################################
# Compare to Von Mises plot                                                   #
###############################################################################
os.makedirs(os.path.dirname(savename), exist_ok=True)
if os.path.isfile(savename):
    alldata = np.load(savename, allow_pickle=True)
else:
    alldata = Parallel(
        n_jobs=n_jobs, verbose=100
    )(delayed(mfunc)(i) for i in tqdm(range(rep)))

    np.save(savename, alldata)

deg_factor = -2*np.log(sc.i1(kappacs) / sc.i0(kappacs))
alldata = np.rad2deg(np.asarray(alldata))
stds = np.rad2deg(np.sqrt(deg_factor))
approx = np.rad2deg(1/np.sqrt(kappacs))
meanvec = np.mean(alldata, axis=0)

fig = plt.figure(figsize=(14, 11))
plt.rcParams.update({'font.size': settings.fs})
spec = fig.add_gridspec(
    ncols=2,
    nrows=2,
    width_ratios=[0.7, 1]
)

ax1 = fig.add_subplot(spec[0, 1])
ax1.plot(
    kappacs, approx, label=r"Large $\kappa_c$ approximation", color="#FC6DAB"
)
ax1.plot(kappacs, meanvec, linewidth=3., label="Mean vector", color="#272838")
ax1.plot(kappacs, stds, label=r"Bessel functions", color="#0ACDFF")

sarg_ang_idx = np.nanargmin(np.abs(meanvec-wide_ang))
approx_ang_idx = np.nanargmin(np.abs(approx-narrow_ang))
approx_ang_idx_55 = np.nanargmin(np.abs(approx-wide_ang))
real_ang_idx = np.nanargmin(np.abs(approx-real_ang))

kappa_sarg_approx = kappacs[sarg_ang_idx]

ax1.scatter(
    kappacs[sarg_ang_idx],
    meanvec[sarg_ang_idx],
    marker="^",
    color="red",
    s=200.,
    zorder=10
)
ax1.scatter(
    kappacs[approx_ang_idx],
    approx[approx_ang_idx],
    marker="*",
    color="red",
    s=300.,
    zorder=10
)
ax1.scatter(
    kappacs[real_ang_idx],
    meanvec[real_ang_idx],
    marker="P",
    color="red",
    s=300.,
    zorder=10
)

ax1.set_xscale("log")
ax1.grid()
ax1.set_xlim(0.1, 100)
ax1.set_ylim(0, 200)
ax1.set_xlabel(
    r"Concentration parameter $\kappa_c$ ($rad^{-2}$)"
)
ax1.set_ylabel(r"Standard deviation ($^\circ$)")
ax1.legend(fontsize=settings.fs*0.7)

###############################################################################
# Parameter search plot                                                       #
###############################################################################
with open(psearch_fname, 'rb') as f:
    gr60s_all, kappas, stds_psearch = pickle.load(f)
K, S = np.meshgrid(np.sqrt(1/kappas)*180/np.pi, stds_psearch)
gr60s = np.nanmedian(gr60s_all, axis=0)

ax2 = fig.add_subplot(spec[1, :])
pcolor_psearch = ax2.pcolormesh(
    K, S, gr60s.T, norm=mpl.colors.LogNorm()
)
gr60s = gaussian_filter(gr60s, 0.9)
CS = ax2.contour(
    K, S, gr60s.T, levels=[2, 10, 50, 200, 400, 600, 800], colors='white'
)
manual_locations = [
    (6, 2),
    (10.5, 7.5),
    (13, 9),
    (16, 12),
    (21, 16),
    (25.5, 16.5),
    (31.5, 19)
]
ax2.clabel(CS, inline=True, fontsize=18, manual=manual_locations)
ax2.scatter(
    np.rad2deg(1/np.sqrt(settings.kappac_i)),
    settings.jitterc_i,
    marker="*",
    color="red",
    s=300.,
    zorder=10,
    clip_on=False
)
ax2.scatter(
    np.rad2deg(1/np.sqrt(kappa_sarg_approx)),
    settings.jitterc_i,
    marker="^",
    color="red",
    s=300.,
    zorder=10,
    clip_on=False
)
ax2.scatter(
    np.rad2deg(1/np.sqrt(4.)),
    settings.jitterc_i,
    marker="P",
    color="red",
    s=300.,
    zorder=10,
    clip_on=False
)


ax2.set_yticks([0, 10, 20, 30], fontsize=settings.fs)
ax2.set_xticks([5, 10, 20, 30, 40, 50, 60], fontsize=settings.fs)
ax2.set_xlabel(
    r'Tuning width $1 / \sqrt{\kappa_c}$ ($^\circ$)', fontsize=settings.fs
)
ax2.set_ylabel(r'Jitter $\sigma_c$ ($^\circ$)', fontsize=settings.fs)
# plt.axis([np.sqrt(1/max(kappas))*180/np.pi, 60, 0, 30])
plt.xlim(5, 60)

div_psearch = make_axes_locatable(ax2)
cax_psearch = div_psearch.append_axes('right', size='4.5%', pad=0.05)
cbar_psearch = fig.colorbar(
    pcolor_psearch,
    cax=cax_psearch,
    fraction=0.020,
    pad=0.04,
    ticks=[1, 10, 100, 1000]
)
cbar_psearch.set_label("Hexasymmetry\n(spk/s)", fontsize=int(settings.fs))

###############################################################################
# Polar plot                                                                  #
###############################################################################
direc = np.linspace(0, 2*np.pi, 360)
kappa_ax3_tuning = 1/np.deg2rad(10)**2
kappa_ax3_doeller = 4.
doeller_ang = np.rad2deg(1 / np.sqrt(kappa_ax3_doeller))
kappa_ax3_sarg = 1.1
sarg_ang = np.rad2deg(1 / np.sqrt(kappa_sarg_approx))
kappa_ax3_narrow = 1/np.deg2rad(narrow_ang)**2

mu_ax3_jitter = 1/np.deg2rad(5)**2

# params visualisation
ax3 = fig.add_subplot(spec[0, 0], projection='polar')
plt.rc('xtick', labelsize=settings.fs)
plt.rc('ytick', labelsize=settings.fs)
ax3.tick_params(axis='both', which='major', labelsize=0.8*settings.fs)
ax3.tick_params(axis='both', which='minor', labelsize=0.8*settings.fs)

wide_tuning = np.zeros(len(direc))
direc = np.linspace(0, 2*np.pi, 360)
for i in range(6):
    axis = i*2*np.pi/6
    # direc = np.linspace(axis - wide, axis + wide, 360)
    wide_tuning += np.exp(
        kappa_ax3_doeller * np.cos(direc - np.pi * i / 3)
    ) / (sc.i0(kappa_ax3_doeller)) / 2 / np.pi
wide_tuning = wide_tuning / np.amax(wide_tuning)
plt.plot(
    direc,
    wide_tuning,
    'blue',
    lw=2,
    linestyle="--"
)

sarg_tuning = np.zeros(len(direc))
direc = np.linspace(0, 2*np.pi, 360)
for i in range(6):
    axis = i*2*np.pi/6
    # direc = np.linspace(axis - wide, axis + wide, 360)
    sarg_tuning += np.exp(
        kappa_sarg_approx * np.cos(direc - np.pi * i / 3)
    ) / (sc.i0(kappa_sarg_approx)) / 2 / np.pi
sarg_tuning = sarg_tuning / np.amax(sarg_tuning)
plt.plot(
    direc,
    sarg_tuning,
    'orange',
    lw=2,
    linestyle="--"
)


narrow_tuning = np.zeros(len(direc))
direc = np.linspace(0, 2*np.pi, 360)
for i in range(6):
    axis = i*2*np.pi/6
    # direc = np.linspace(axis - wide, axis + wide, 360)
    narrow_tuning += np.exp(
        kappa_ax3_narrow * np.cos(direc - np.pi * i / 3)
    ) / (sc.i0(kappa_ax3_narrow)) / 2 / np.pi
narrow_tuning = narrow_tuning / np.amax(narrow_tuning)
plt.plot(
    direc,
    narrow_tuning,
    'green',
    lw=2,
    linestyle="--"
)

direc = np.linspace(0, 2*np.pi, 360)
tuning_width = np.zeros(len(direc))

for i in range(6):
    plt.arrow(
        i*np.pi/3,
        0,
        0,
        np.amax(wide_tuning),
        color='r',
        length_includes_head=True,
        head_width=0.15,
        head_length=0.20,
        zorder=5
    )

ax3.set_rlabel_position(-22.5)
ax3.set_xticks(np.pi/180.*np.array([0, 90, 180, 270]))
ax3.set_rticks(
    [np.amax(wide_tuning)], []
)
ax3.set_ylim([0, 1.2])
ax3.grid(True)
ax3.text(
    1,
    0.7,
    r'$\frac{1}{\sqrt{\kappa_{c}}}$' + f"={doeller_ang:.0f}$^o$",
    fontsize=settings.fs*1.1,
    transform=ax3.transAxes,
    color="blue"
)
ax3.text(
    1,
    0.9,
    r'$\frac{1}{\sqrt{\kappa_{c}}}$' + f"={sarg_ang:.0f}$^o$",
    fontsize=settings.fs*1.1,
    transform=ax3.transAxes,
    color="darkorange"
)
ax3.text(
    1,
    0.3,
    r'$\frac{1}{\sqrt{\kappa_{c}}}$' + f"={int(narrow_ang)}$^o$",
    fontsize=settings.fs*1.1,
    transform=ax3.transAxes,
    color="green"
)

plt.subplots_adjust(
    wspace=0.3,
    hspace=0.3
)

ax_pos(ax2, -0.05, -0.02, 0.675, 1.)
ax_pos(ax3, -0.025, 0.02, 0.95, 0.95)
ax_pos(ax1, -0.01, 0.02, 0.6, 1.)

plot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "extend_params",
    "Figure_tuning_widths.png"
)
os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(plot_fname, dpi=300)
