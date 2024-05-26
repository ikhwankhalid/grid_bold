import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
import settings
import matplotlib as mpl
from math import isinf
from utils.grid_funcs import gen_offsets, gridpop_conj, traj, traj_pwl
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils.utils import (
    ax_pos,
    convert_to_rhombus,
    get_hexsym,
    grid_2d,
    grid_meanfr
)

mpl.rcParams['figure.dpi'] = 300
plt.box(False)

###############################################################################
# Parameters                                                                  #
###############################################################################
fs, meanoff, imax = settings.fs, settings.meanoff, 10

ox, oy = gen_offsets(N=settings.N)
oxr, oyr = convert_to_rhombus(ox, oy)

bins, phbins, amax = settings.bins, settings.phbins, settings.amax
tmax_pwl, dt_pwl, dt_pwl2, dphi_pwl, part = 72000, 9, 0.1, 2., 4000

phis = np.linspace(0, 2 * np.pi, settings.phbins, endpoint=False)
tmax_rw, tmax_star = 600, 30

analytic_colour, hextextloc = "#ff42ef", (0.6, 1.1)

plot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "figure3_conjunctive.png"
)

fig = plt.figure(figsize=(20, 12))
spec = fig.add_gridspec(
    ncols=4,
    nrows=6,
    width_ratios=[1, 1.5, 1.5, 0.4],
    height_ratios=[1, 1, 1, 1, 1, 1]
)


###############################################################################
# Preferred direction and parameter visualisation                             #
###############################################################################
direc = np.linspace(0, 2*np.pi, 360)
# settings.kappac_i = 1/np.deg2rad(10)**2
mu_ax2_jitter = 1/np.deg2rad(5)**2

widecolor = "#ff6000"

# preferred direction
ax = fig.add_subplot(spec[0:3, 0])
X, Y = np.meshgrid(np.linspace(-50, 50, 1000), np.linspace(-50, 50, 1000))
gr = amax*grid_2d(X, Y, grsc=30, angle=0, offs=np.array([0, 0]))
plt.pcolormesh(X, Y, gr, vmin=0, vmax=8, shading='auto')
plt.arrow(
    0,
    0,
    15,
    np.sqrt(3)/2*30,
    color='r',
    length_includes_head=True,
    head_width=5
)
ax.set_aspect('equal')
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)
plt.xticks([])
plt.yticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
c = plt.colorbar(cax=cax, ticks=[0, 4, 8])
c.set_label('Firing rate (spk/s)', fontsize=fs)
c.ax.tick_params(labelsize=fs)


def conv_circ(signal, ker):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return np.real(np.fft.ifft(np.fft.fft(signal)*np.fft.fft(ker)))


def get_tuning_conv(direc, mu, kappa):
    idx = np.argmin(np.abs(direc - np.pi/3))

    mu_denom = (sc.i0(mu)) * 2 * np.pi
    if not isinf(mu_denom):
        tuning_jitter = np.exp(
            mu * np.cos(direc - np.pi / 3)
        ) / (sc.i0(mu)) / 2 / np.pi
    else:
        tuning_jitter = np.zeros(len(direc))
        tuning_jitter[idx] = 1

    kappa_denom = (sc.i0(kappa)) * 2 * np.pi
    if not isinf(kappa_denom):
        tuning_width = np.exp(
            kappa * np.cos(direc)
        ) / (sc.i0(kappa)) / 2 / np.pi
    else:
        tuning_width = np.zeros(len(direc))
        tuning_width[idx] = 1

    conv = conv_circ(tuning_jitter, tuning_width)
    return conv


# visualization of tuning width
ax2 = fig.add_subplot(spec[0:3, 1], projection='polar')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=0.8*fs)
ax2.tick_params(axis='both', which='minor', labelsize=0.8*fs)

wide_tuning = np.zeros(len(direc))
direc = np.linspace(0, 2*np.pi, 360)
for i in range(6):
    axis = i*2*np.pi/6
    # wide = np.deg2rad(30)
    # direc = np.linspace(axis - wide, axis + wide, 360)
    wide_tuning += np.exp(
        settings.kappac_r * np.cos(direc - np.pi * i / 3)
    ) / (sc.i0(settings.kappac_r)) / 2 / np.pi
wide_tuning = wide_tuning / np.amax(wide_tuning)
plt.plot(
    direc,
    wide_tuning,
    widecolor,
    lw=2,
    linestyle="--"
)

direc = np.linspace(0, 2*np.pi, 360)
tuning_width = np.zeros(len(direc))
for i in range(6):
    tuning_width = np.exp(
        settings.kappac_i * np.cos(direc - np.pi * i / 3)
    ) / (sc.i0(settings.kappac_i)) / 2 / np.pi
    normed_width = tuning_width / np.amax(tuning_width)
    plt.plot(
        direc,
        normed_width,
        'blue',
        lw=2,
        linestyle="--"
    )

for i in range(6):
    plt.arrow(
        i*np.pi/3,
        0,
        0,
        np.amax(normed_width),
        color='r',
        length_includes_head=True,
        head_width=0.15,
        head_length=0.20,
        zorder=5
    )

ax2.set_rlabel_position(-22.5)
ax2.set_xticks(np.pi/180.*np.array([0, 90, 180, 270]))
ax2.set_rticks(
    []
)
ax2.set_ylim([0, 1.03])
ax2.grid(True)

# convolution of jitter and tuning width
ax2b = fig.add_subplot(spec[0:3, 2], projection='polar')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
ax2b.tick_params(axis='both', which='major', labelsize=0.8*fs)
ax2b.tick_params(axis='both', which='minor', labelsize=0.8*fs)

direc = np.linspace(0, 2*np.pi, 360)

tuning_jitter = np.exp(
    mu_ax2_jitter * np.cos(direc - np.pi / 3)
) / (sc.i0(mu_ax2_jitter)) / 2 / np.pi
tuning_width = np.exp(
    settings.kappac_i * np.cos(direc - np.pi / 3)
) / (sc.i0(settings.kappac_i)) / 2 / np.pi
wide_tuning = np.exp(
    settings.kappac_r * np.cos(direc - np.pi / 3)
) / (sc.i0(settings.kappac_r)) / 2 / np.pi
r = get_tuning_conv(direc, mu_ax2_jitter, settings.kappac_i)

plt.plot(
    direc,
    tuning_jitter / np.amax(tuning_jitter),
    'green',
    lw=1.5,
    linestyle="--"
)
# plt.plot(
#     direc,
#     wide_tuning / np.amax(wide_tuning),
#     widecolor,
#     lw=1.5,
#     linestyle="--"
# )
plt.plot(
    direc,
    tuning_width / np.amax(tuning_width),
    'blue',
    lw=1.5,
    linestyle="--"
)
# plt.plot(
#     direc, r/np.amax(r), 'k', lw=1., alpha=0.4
# )
plt.fill(direc, r/np.amax(r), alpha=0.4, color="gray")

plt.arrow(
    np.pi/3,
    0,
    0,
    1.,
    color='r',
    length_includes_head=True,
    head_width=0.15,
    head_length=0.20,
    zorder=5
)
ax2b.set_rlabel_position(-22.5)
ax2b.set_xticks(np.pi/180.*np.array([0, 90, 180, 270]))
ax2b.set_rticks(
    [np.amax(tuning_width) / 2, np.amax(tuning_width)],
    []
)
ax2b.set_ylim([0, 1.03])
ax2b.set_xticks(np.pi/180.*np.array([0, 90, 180, 270]))
ax2b.grid(True)
plt.text(
    0.45,
    0.6,
    r"$\frac{1}{\sqrt{\kappa_{c}}} = $" +
    f"{np.rad2deg(np.sqrt(1/settings.kappac_i)):.0f}" + r"$^\circ$",
    fontsize=settings.fs*1.15,
    transform=plt.gcf().transFigure,
    color="blue",
    # weight="bold"
)
plt.text(
    0.45,
    0.55,
    r"$\frac{1}{\sqrt{\kappa_{c}}} = $" +
    f"{np.rad2deg(np.sqrt(1/settings.kappac_r)):.0f}" + r"$^\circ$",
    fontsize=settings.fs*1.15,
    transform=plt.gcf().transFigure,
    color=widecolor,
    # weight="bold"
)
plt.text(
    0.6,
    0.575,
    r'$\sigma_c = $' + f"{5}" + r'$^\circ$',
    fontsize=settings.fs*1.15,
    transform=plt.gcf().transFigure,
    color="green",
    # weight="bold"
)


###############################################################################
# Parameter search                                                            #
###############################################################################
# conjunctive parameter search
print("Doing parameter search...")

nn = 40


def conj_paramsearch(nn):
    n_kappas = nn * 3/2 + 1
    n_sigmas = nn + 1
    widthmax = 60
    sigmamax = 20
    widths = np.linspace(1e-2, widthmax, n_kappas)
    kappas = (180 / np.pi / widths)**2
    stds = np.linspace(0, sigmamax, n_sigmas)
    gr60s = np.zeros((n_kappas, n_sigmas))

    def inner_loop(i, width, stds):
        row = np.zeros(len(stds))
        for j, std in enumerate(stds):
            ox, oy = gen_offsets(N=settings.N, kappacl=0.)
            oxr, oyr = convert_to_rhombus(ox, oy)
            trajec = traj(
                settings.dt,
                tmax_rw,
                sp=settings.speed,
                dphi=settings.dphi
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

    gr60s = Parallel(n_jobs=-1)(
        delayed(inner_loop)(i, width, stds) for i, width in enumerate(widths)
    )
    gr60s = np.array(gr60s)

    return gr60s, kappas, stds


# psearch_fname = os.path.join(
#     settings.loc,
#     "conjunctive",
#     "fig3",
#     "psearch.pkl"
# )

psearch_fname = os.path.join(
    settings.loc,
    "conjunctive",
    # "extend_psearch",
    "extend_psearch_star",
    "psearch_30sigma.pkl"
)


ax9 = fig.add_subplot(spec[0:3, 3], adjustable='box', aspect=1)
if not os.path.isfile(psearch_fname):
    gr60s, kappas, stds = conj_paramsearch(nn)
    os.makedirs(os.path.dirname(psearch_fname), exist_ok=True)
    with open(psearch_fname, 'wb') as f:
        pickle.dump([gr60s, kappas, stds], f)
else:
    with open(psearch_fname, 'rb') as f:
        gr60s, kappas, stds = pickle.load(f)


K, S = np.meshgrid(np.sqrt(1/kappas)*180/np.pi, stds)
gr60s = np.nanmedian(gr60s, axis=0)
# pcolor_psearch = plt.imshow(
#     gr60s.T, extent=(0, 60, 0, 30), origin="lower", aspect=1,
#     norm=matplotlib.colors.LogNorm()
# )
pcolor_psearch = plt.pcolormesh(
    K, S, gr60s.T, norm=matplotlib.colors.LogNorm()
)
gr60s = gaussian_filter(gr60s, settings.smooth_sigma)
CS = plt.contour(
    K,
    S,
    gr60s.T,
    levels=[2, 10, 50, 200, 800],
    colors='white'
)
manual_locations = [(9, 5), (15, 11), (20, 15), (26, 15), (33, 18)]
cont_labels = ax9.clabel(CS, inline=True, fontsize=15, manual=manual_locations)
for label in cont_labels:
    label.set_rotation(0)
    label.set_fontweight("bold")
# plt.xticks([5, 15, 25, 35], fontsize=fs)
plt.xticks([5, 10, 20, 30, 40], fontsize=fs)
# plt.yticks([0, 5, 10], fontsize=fs)
plt.yticks([0, 10, 20, 30], fontsize=fs)
plt.xlabel(r'Tuning width $1 / \sqrt{\kappa_c}$ ($^\circ$)', fontsize=fs)
plt.ylabel(r'Jitter $\sigma_c$ ($^\circ$)', fontsize=fs)
plt.scatter(
    np.rad2deg(np.sqrt(1/settings.kappac_r)),
    3.,
    c='red',
    s=150,
    marker="*",
    zorder=10
)
plt.scatter(
    np.rad2deg(np.sqrt(1/10.)), 1.5, c='red', s=150, marker="^", zorder=10
)
clip = plt.scatter(
    np.rad2deg(np.sqrt(1/settings.kappac_i)), 0, c='red',
    s=150,
    marker="o",
    zorder=10
)
clip.set_clip_on(False)
plt.axis([np.sqrt(1/max(kappas))*180/np.pi, 30, 0, 30])
plt.xlim(5, 40)

# plt.savefig(plot_fname,dpi=300)


###############################################################################
# Vary conjunctive cell paremeters                                            #
###############################################################################
trajec = traj(
    settings.dt,
    settings.tmax,
    sp=settings.speed,
    dphi=settings.dphi
)
ox, oy = gen_offsets(N=settings.N, kappacl=0.)
oxr, oyr = convert_to_rhombus(ox, oy)


# kappa = 50
print("Comparing params...")
ax3 = fig.add_subplot(spec[3, 2])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
mfrk50j0p10_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "mfrk50j0p10.pkl"
)
mfrk50j0p03_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "mfrk50j0p03.pkl"
)
mfrk50j0_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            (
                "conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma0_kappa50"
                "_ratio1.npy"
            )
        )
)
if not os.path.isfile(mfrk50j0p10_nfname):
    fr_mean_k50j0p10, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )[1::2]
    gr60_k50j0p10 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(mfrk50j0p10_nfname), exist_ok=True)
    with open(mfrk50j0p10_nfname, 'wb') as f:
        pickle.dump((fr_mean_k50j0p10, gr60_k50j0p10), f)
else:
    with open(mfrk50j0p10_nfname, 'rb') as f:
        fr_mean_k50j0p10, gr60_k50j0p10 = pickle.load(f)

if not os.path.isfile(mfrk50j0p03_nfname):
    fr_mean_k50j0p03, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_r,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )[1::2]
    gr60_k50j0p03 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(mfrk50j0p03_nfname), exist_ok=True)
    with open(mfrk50j0p03_nfname, 'wb') as f:
        pickle.dump((fr_mean_k50j0p03, gr60_k50j0p03), f)
else:
    with open(mfrk50j0p03_nfname, 'rb') as f:
        fr_mean_k50j0p03, gr60_k50j0p03 = pickle.load(f)

with open(mfrk50j0_afname, 'rb') as f:
    amfr_conjk50j0 = np.load(f)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    fr_mean_k50j0p10,
    'k',
    linewidth=3,
    alpha=1.,
    zorder=1
)
# plt.plot(
#     np.linspace(0, 360, settings.phbins),
#     fr_mean_k50j0p03,
#     'k',
#     linestyle="--",
#     linewidth=3,
#     alpha=1.,
#     zorder=1
# )
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_conjk50j0,
    color=analytic_colour,
    linestyle='dashed',
    linewidth=1.5,
    alpha=1.,
    zorder=10
)
clip = plt.scatter(380, 2500, c='red', s=100, marker="o")
clip.set_clip_on(False)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [])
plt.yticks(fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60_k50j0p10, 1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax3.transAxes
)
plt.ylim([0, 5100])


# def conv_circ(signal, ker):
#     '''
#         signal: real 1D array
#         ker: real 1D array
#         signal and ker must have same shape
#     '''
#     return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

# def get_tuning_conv(direc, mu, kappa):
#     tuning_jitter = np.exp(mu * np.cos(direc - np.pi / 3)) / (sc.i0(mu))
#     tuning_width = np.exp(kappa * np.cos(direc)) / (sc.i0(kappa))
#     conv = conv_circ(tuning_jitter, tuning_width)
#     return conv


# params visualisation
ax3_tuning = fig.add_subplot(spec[3, 3], projection='polar')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
ax3_tuning.tick_params(axis='both', which='major', labelsize=0.8*fs)
ax3_tuning.tick_params(axis='both', which='minor', labelsize=0.8*fs)
# mu_ax3_tuning = 1/np.sqrt((settings.jitterc_i + 1e-2) * np.pi / 180)
mu_ax3_tuning = 1/(np.deg2rad(settings.jitterc_i+1e-2))**2
kappa_ax3_tuning = settings.kappac_i
direc = np.linspace(0, 2*np.pi, 360)
r = get_tuning_conv(direc, mu_ax3_tuning, kappa_ax3_tuning)
# plt.plot(direc, r/np.mean(r)*np.mean(gr), 'k', lw=2)
plt.fill(direc, r/np.mean(r)*np.mean(gr), alpha=0.4, color="gray")
plt.arrow(
    np.pi/3,
    0,
    0,
    np.amax(r/np.mean(r)*np.mean(gr)),
    color='r',
    length_includes_head=True,
    head_width=0.1,
    head_length=1,
    zorder=5
)
# ax3_tuning.vlines(np.pi/3, 0, 1, zorder=10)
ax3_tuning.set_rticks(
    []
)
ax3_tuning.set_ylim([0, np.amax(r/np.mean(r)*np.mean(gr)) * 1.03])
ax3_tuning.set_rlabel_position(-22.5)
ax3_tuning.set_xticks([])
ax3_tuning.grid(True)


# kappa = 10 ##################################################################
ax4 = fig.add_subplot(spec[4, 2])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
mfrk10j15p10_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "mfrk10j15p10.pkl"
)
mfrk10j15p03_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "mfrk10j15p03.pkl"
)
conjk25j5_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "conj_kappa10_sigma1_5.npy"
        )
)
if not os.path.isfile(mfrk10j15p10_nfname):
    direc_binned, fr_mean_k10j15p10, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=10.,
        jitter=1.5
    )
    gr60_k10j15p10 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(mfrk10j15p10_nfname), exist_ok=True)
    with open(mfrk10j15p10_nfname, 'wb') as f:
        pickle.dump((fr_mean_k10j15p10, gr60_k10j15p10), f)
else:
    with open(mfrk10j15p10_nfname, 'rb') as f:
        fr_mean_k10j15p10, gr60_k10j15p10 = pickle.load(f)
with open(conjk25j5_afname, 'rb') as f:
    amfr_conjk25j5 = np.load(f)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    fr_mean_k10j15p10,
    'k',
    linewidth=3,
    alpha=1.,
    zorder=1
)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_conjk25j5,
    color=analytic_colour,
    linestyle='dashed',
    linewidth=1.5,
    alpha=1.,
    zorder=10
)
clip = plt.scatter(380, 3000, c='red', s=100, marker="^")
clip.set_clip_on(False)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [])
plt.yticks(fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60_k10j15p10, 1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax4.transAxes
)
plt.ylim([0, 5100])


# params visualisation
ax4_tuning = fig.add_subplot(spec[4, 3], projection='polar')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
ax4_tuning.tick_params(axis='both', which='major', labelsize=0.8*fs)
ax4_tuning.tick_params(axis='both', which='minor', labelsize=0.8*fs)
# mu_ax4_tuning = 1/np.sqrt(1.5 * np.pi / 180)
mu_ax4_tuning = 1/(np.deg2rad(1.5))**2
kappa_ax4_tuning = 10.
direc = np.linspace(0, 2*np.pi, 360)
r = get_tuning_conv(direc, mu_ax4_tuning, kappa_ax4_tuning)
# plt.plot(direc, r/np.mean(r)*np.mean(gr), 'k', lw=2)
plt.fill(direc, r/np.mean(r)*np.mean(gr), alpha=0.4, color="gray")
# plt.plot(direc,r/np.mean(r)*np.mean(gr),'k',lw=2)
plt.arrow(
    np.pi/3,
    0,
    0,
    np.amax(r/np.mean(r)*np.mean(gr)),
    color='r',
    length_includes_head=True,
    head_width=0.1,
    head_length=1,
    zorder=5
)
ax4_tuning.set_rticks(
    []
)
ax4_tuning.set_ylim([0, np.amax(r/np.mean(r)*np.mean(gr)) * 1.03])
ax4_tuning.set_rlabel_position(-22.5)
ax4_tuning.set_xticks([])
ax4_tuning.grid(True)


# kappa = 4 ###################################################################
ax5 = fig.add_subplot(spec[5, 2])
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
frdata_rpropi_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata_rpropi.pkl"
)
frdata_allr_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata_allr.pkl"
)
conjk2p5j10_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "conj_kappa4_sigma3.npy"
        )
)
if not os.path.isfile(frdata_rpropi_fname):
    direc_binned, fr_mean_rpropi, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_r,
        jitter=settings.jitterc_r
    )
    gr60_rpropi = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(frdata_rpropi_fname), exist_ok=True)
    with open(frdata_rpropi_fname, 'wb') as f:
        pickle.dump((fr_mean_rpropi, gr60_rpropi), f)
else:
    with open(frdata_rpropi_fname, 'rb') as f:
        fr_mean_rpropi, gr60_rpropi = pickle.load(f)
if not os.path.isfile(frdata_allr_fname):
    direc_binned, fr_mean_allr, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_r,
        kappa=settings.kappac_r,
        jitter=settings.jitterc_r
    )
    gr60_allr = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(frdata_allr_fname), exist_ok=True)
    with open(frdata_allr_fname, 'wb') as f:
        pickle.dump((fr_mean_allr, gr60_allr), f)
else:
    with open(frdata_allr_fname, 'rb') as f:
        fr_mean_allr, gr60_allr = pickle.load(f)
with open(conjk2p5j10_afname, 'rb') as f:
    amfr_conjk2p5j10 = np.load(f)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    fr_mean_rpropi,
    'k',
    linewidth=3,
    alpha=1.,
    zorder=1
)
# plt.plot(
#     np.linspace(0, 360, settings.phbins),
#     fr_mean_allr,
#     'blue',
#     linestyle="--",
#     linewidth=3,
#     alpha=1.,
#     zorder=1
# )
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_conjk2p5j10,
    color=analytic_colour,
    linestyle='dashed',
    linewidth=1.5,
    alpha=1.,
    zorder=10
)
clip = plt.scatter(380, 2500, c='red', s=100, marker="*")
clip.set_clip_on(False)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel(r'Movement direction ($^\circ$)', fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60_rpropi, 1))+" spk/s" +
    ' ('+str(np.round(gr60_allr, 1))+" spk/s)",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax5.transAxes
)
plt.ylim([0, 5100])


# params visualisation
ax5_tuning = fig.add_subplot(spec[5, 3], projection='polar')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
ax5_tuning.tick_params(axis='both', which='major', labelsize=0.8*fs)
ax5_tuning.tick_params(axis='both', which='minor', labelsize=0.8*fs)
# mu_ax5_tuning = 1/np.sqrt(settings.jitterc_r * np.pi / 180)
mu_ax5_tuning = 1/(np.deg2rad(settings.jitterc_r))**2
kappa_ax5_tuning = settings.kappac_r
direc = np.linspace(0, 2*np.pi, 360)
r = get_tuning_conv(direc, mu_ax5_tuning, kappa_ax5_tuning)
# plt.plot(direc, r/np.mean(r)*np.mean(gr), 'k', lw=2)
plt.fill(direc, r/np.mean(r)*np.mean(gr), alpha=0.4, color="gray")
# plt.plot(direc,r/np.mean(r)*np.mean(gr),'k',lw=2)
plt.arrow(
    np.pi/3,
    0,
    0,
    np.amax(r/np.mean(r)*np.mean(gr)),
    color='r',
    length_includes_head=True,
    head_width=0.1,
    head_length=1,
    zorder=5
)
ax5_tuning.set_rticks(
    []
)
ax5_tuning.set_ylim([0, np.amax(r/np.mean(r)*np.mean(gr)) * 1.03])
ax5_tuning.set_rlabel_position(-22.5)
ax5_tuning.set_xticks([])
ax5_tuning.grid(True)


###############################################################################
# Vary trajectory type                                                        #
###############################################################################
trajc = "w"
pointc = "r"


pwl_traj_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "pwl_traj.pkl"
)
rw_traj_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "rw_traj.pkl"
)


# starlike walk
# star-like run trajectory
ax_star_traj = fig.add_subplot(spec[3, 0], adjustable='box', aspect="auto")
plt.plot([0, 0], [-90, 90], trajc, linewidth=1.5)
plt.plot([-90, 90], [0, 0], trajc, linewidth=1.5)
plt.plot(
    [-1/np.sqrt(2) * 90, 1/np.sqrt(2) * 90],
    [-1/np.sqrt(2) * 90, 1/np.sqrt(2) * 90],
    trajc,
    linewidth=1.5
)
plt.plot(
    [-1/np.sqrt(2) * 90, 1/np.sqrt(2) * 90],
    [1/np.sqrt(2) * 90, -1/np.sqrt(2) * 90],
    trajc
)
ax_star_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
X_bgr, Y_bgr, _ = np.meshgrid(
    np.linspace(-120, 120, 1000),
    np.linspace(-120, 120, 1000),
    1
)
gr_bgr = grid_meanfr(X_bgr, Y_bgr, grsc=30, angle=0, offs=np.array([0, 0]))
ax_star_traj.yaxis.set_ticks_position("right")
ax_star_traj.pcolormesh(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)


bar_dist = 4 * settings.grsc
bar_ang = -120
bar_len = 2 * settings.grsc
bar_off = settings.grsc

x_center = bar_dist * np.cos(2 * np.pi * bar_ang / 360)
y_center = bar_dist * np.sin(2 * np.pi * bar_ang / 360)

x_positions = [
    x_center + bar_off,
    x_center - bar_len + bar_off,
    x_center + bar_len + bar_off
]
y_positions = [y_center, y_center, y_center]

plt.plot(x_positions, y_positions, color="red", lw=3)

plt.axis("square")

ax6 = fig.add_subplot(spec[3, 1], adjustable='box', aspect="auto")
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
# ax6.spines['bottom'].set_visible(False)
# ax6.spines['left'].set_visible(False)
ax6.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
stardata_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "stardata.pkl"
)
conjstar_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            # "conjunctive_sw_N1024_nphi360_rmin0_rmax3.npy"
            "clustering_sw_N1024_nphi360_rmin0_rmax3.npy"
        )
)


if not os.path.isfile(stardata_fname):
    r, phi, indoff = np.meshgrid(
        np.linspace(0, settings.rmax, bins),
        np.linspace(0, 2 * np.pi, phbins, endpoint=False),
        np.arange(len(oxr))
    )
    X, Y = r * np.cos(phi), r*np.sin(phi)
    grids = amax * grid_meanfr(X, Y, offs=(oxr, oyr))
    grids2 = grids.copy()

    Nconj = int(settings.propconj_i*settings.N)
    mu = np.mod(
        np.random.randint(0, 6, Nconj) * 60 + 0*np.random.randn(Nconj), 360
    )
    vonmi = np.exp(
        settings.kappac_i*np.cos(
            np.pi/180*(np.linspace(0, 360, phbins)[:, None]-mu[None, :])
        )
    ) / sc.i0(settings.kappac_i)
    # change the first Nconj cells
    grids[:, :, :Nconj] = grids[:, :, :Nconj]*vonmi[:, None, :]

    fr_mean_base = np.sum(np.sum(grids2, axis=1)/bins, axis=1)
    fr_mean = np.sum(np.sum(grids, axis=1)/bins, axis=1)
    fr_mean = fr_mean/np.mean(fr_mean)*np.mean(fr_mean_base)

    gr2 = np.sum(grids, axis=2)
    gr60 = np.abs(np.sum(gr2 * np.exp(-6j*phi[:, :, 0])))/np.size(gr2)
    gr60_path = np.abs(np.sum(np.exp(-6j*phi[:, :, 0])))/np.size(gr2)
    gr0 = np.sum(gr2)/np.size(gr2)

    os.makedirs(os.path.dirname(stardata_fname), exist_ok=True)
    with open(stardata_fname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(stardata_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjstar_afname, 'rb') as f:
    amfr_conjstar = np.load(f)


plt.plot(
    np.linspace(0, 360, settings.phbins),
    fr_mean,
    'k',
    linewidth=3,
    alpha=1.,
    zorder=1
)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_conjstar,
    color=analytic_colour,
    linestyle='dashed',
    linewidth=1.5,
    alpha=1.,
    zorder=10
)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [])
plt.yticks(fontsize=fs)
# plt.xlabel('Movement direction (deg)',fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
# plt.title('star-like walk',fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60, 1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax6.transAxes
)
# plt.ylim([0,1.2*max(fr_mean)])
plt.ylim([0, 5100])


# piecewise linear trajectory
# with open(pwl_traj_fname, 'rb') as f:
#         trajec = pickle.load(f)


if not os.path.isfile(pwl_traj_fname):
    trajec = traj_pwl(
        settings.phbins_pwl,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )
    os.makedirs(os.path.dirname(pwl_traj_fname), exist_ok=True)
    with open(pwl_traj_fname, 'wb') as f:
        pickle.dump(trajec, f)
else:
    with open(pwl_traj_fname, 'rb') as f:
        trajec = pickle.load(f)


ax_pwl_traj = fig.add_subplot(spec[4, 0], adjustable='box', aspect="auto")
plt.plot(
    trajec[1][:int(part*2)], trajec[2][:int(part*2)], trajc, linewidth=1.5
)
# plt.plot(trajec[1],trajec[2],'k')
ax_pwl_traj.set_aspect('equal', adjustable='box')
plt.axis('square')
plt.xticks([])
plt.yticks([])
# plt.xticks([0])
# plt.yticks([0])


bar_dist = 11 * settings.grsc
bar_ang = np.deg2rad(60)
plt.plot(
    [
        bar_dist * np.cos(bar_ang), bar_dist * np.cos(bar_ang) - bar_len,
        bar_dist * np.cos(bar_ang) + bar_len
    ],
    [
        bar_dist * np.sin(bar_ang), bar_dist * np.sin(bar_ang),
        bar_dist * np.sin(bar_ang)
    ],
    color="red",
    lw=3
)


X_bgr, Y_bgr, _ = np.meshgrid(
    np.linspace(-2000, 2000, 4000),
    np.linspace(-2000, 2000, 4000),
    1
)
gr_bgr = grid_meanfr(X_bgr, Y_bgr, grsc=30, angle=0, offs=np.array([0, 0]))
ax_pwl_traj.pcolormesh(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)


# pwl firing rate
ax7 = fig.add_subplot(spec[4, 1], adjustable='box', aspect="auto")
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
# ax7.spines['bottom'].set_visible(False)
# ax7.spines['left'].set_visible(False)
ax7.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
pwldata_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "pwldata.pkl"
)
conjpwl_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            (
                "conjunctive_pwl_N1024_nphi360_rmin0_rmax3_sigma0_kappa50"
                "_ratio1.npy"
            )
        )
)
if not os.path.isfile(pwldata_fname):
    t, x, y, direc = traj_pwl(
        settings.phbins_pwl,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(pwldata_fname), exist_ok=True)
    with open(pwldata_fname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(pwldata_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjpwl_afname, 'rb') as f:
    amfr_conjpwl = np.load(f)


plt.plot(
    np.linspace(0, 360, settings.phbins),
    fr_mean,
    'k',
    linewidth=3,
    alpha=1.,
    zorder=1
)
plt.plot(
    np.linspace(0, 360, settings.phbins_pwl),
    # np.linspace(0,360,len(amfr_conjpwl)),
    amfr_conjpwl,
    color=analytic_colour,
    linestyle='dashed',
    linewidth=1.5,
    alpha=1.,
    zorder=10
)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [])
plt.yticks(fontsize=fs)
# plt.xlabel('Movement direction (deg)',fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
# plt.title('piece-wise linear walk',fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60, 1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax7.transAxes
)
# plt.ylim([0,1.2*max(fr_mean)])
plt.ylim([0, 5100])


# random walk trajectory
if not os.path.isfile(rw_traj_fname):
    trajec = traj(
        settings.dt,
        settings.tmax,
        sp=settings.speed,
        dphi=settings.dphi
    )
    os.makedirs(os.path.dirname(rw_traj_fname), exist_ok=True)
    with open(rw_traj_fname, 'wb') as f:
        pickle.dump(trajec, f)
else:
    with open(rw_traj_fname, 'rb') as f:
        trajec = pickle.load(f)


ax_rw_traj = fig.add_subplot(spec[5, 0], adjustable='box', aspect="auto")
plt.plot(trajec[1][:part], trajec[2][:part], trajc, linewidth=1.5)
ax_rw_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
plt.axis('square')


bar_dist = 1 * settings.grsc
bar_ang = np.deg2rad(60)
bar_len = 4 * settings.grsc
plt.plot(
    [
        bar_dist * np.cos(bar_ang) + 2 * settings.grsc,
        bar_dist * np.cos(bar_ang) + bar_len + 2 * settings.grsc
    ],
    [
        bar_dist * np.sin(bar_ang), bar_dist * np.sin(bar_ang)
    ],
    color="red",
    lw=3
)


xtr, ytr, dirtr = trajec[1][:part], trajec[2][:part], trajec[3][:part]
tol = 5 * np.pi / 180
ax_rw_traj.pcolormesh(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)


# rw firing rate
rwdata_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "rwdata.pkl"
)
conjrw_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            (
                "2D_conjunctive_rw_N1024_nphi360_T180000_dt0_01_v0_1"
                "_sigmatheta0_5_sigma0_kappa50_ratio1.npy"
            )
        )
)
ax8 = fig.add_subplot(spec[5, 1], adjustable='box', aspect="auto")
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
if not os.path.isfile(rwdata_fname):
    trajec = traj(
        settings.dt,
        settings.tmax,
        sp=settings.speed,
        dphi=settings.dphi
    )
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(rwdata_fname), exist_ok=True)
    with open(rwdata_fname, 'wb') as f:
        pickle.dump([fr_mean, gr60], f)
else:
    with open(rwdata_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjrw_afname, 'rb') as f:
    amfr_conjrw = np.load(f)


plt.plot(
    np.linspace(0, 360, settings.phbins),
    fr_mean,
    'k',
    linewidth=3,
    alpha=1.,
    zorder=1
)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_conjrw,
    color=analytic_colour,
    linestyle='dashed',
    linewidth=1.5,
    alpha=1.,
    zorder=10
)

plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel(r'Movement direction ($^\circ$)', fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)

# Adding a text label with hexasymmetry value
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60, 1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax8.transAxes
)
# plt.ylim([0,1.2*max(fr_mean)])
plt.ylim([0, 5100])


###############################################################################
# Tweaking figure layout and saving                                           #
###############################################################################
plt.subplots_adjust(wspace=0.60, hspace=1.5)

ax_pos(ax, 0.03, -0.01, 1.1, 1.1)
ax_pos(ax9, -0.105, 0, 3.7, 2.7)
tuning_axs = [ax3_tuning, ax4_tuning, ax5_tuning]
for ax_tuning in tuning_axs:
    ax_pos(ax_tuning, 0, 0, 2, 2)
div_psearch = make_axes_locatable(ax9)
cax_psearch = div_psearch.append_axes('right', size='4.5%', pad=0.05)
cbar_psearch = fig.colorbar(
    pcolor_psearch,
    cax=cax_psearch,
    fraction=0.020,
    pad=0.04,
    ticks=[1, 10, 100, 1000]
)
cbar_psearch.set_label("Hexasymmetry\n(spk/s)", fontsize=int(settings.fs))


fr_axs = [ax3, ax4, ax5]
for fr_ax in fr_axs:
    ax_pos(fr_ax, 0.05, 0, 1.1, 1.)

fr_axs = [ax6, ax7, ax8]
for fr_ax in fr_axs:
    ax_pos(fr_ax, 0.012, 0, 1.1, 1.)


# make trajectory plots larger
posstar = ax_star_traj.get_position()
pospwl = ax_pwl_traj.get_position()
posrw = ax_rw_traj.get_position()

pointsstar = posstar.get_points()
pointspwl = pospwl.get_points()
pointsrw = posrw.get_points()

mean_pointsstar = np.array(
    [(pointsstar[0][0] + pointsstar[1][0])/2,
        (pointsstar[0][1] + pointsstar[1][1])/2]
)
mean_pointspwl = np.array(
    [(pointspwl[0][0] + pointspwl[1][0])/2,
        (pointspwl[0][1] + pointspwl[1][1])/2]
)
mean_pointsrw = np.array(
    [(pointsrw[0][0] + pointsrw[1][0])/2,
        (pointsrw[0][1] + pointsrw[1][1])/2]
)

pointsstar -= mean_pointsstar
pointspwl -= mean_pointspwl
pointsrw -= mean_pointsrw

pointsstar = 2.3*pointsstar
pointspwl = 2.3*pointspwl
pointsrw = 2.3*pointsrw

pointsstar += mean_pointsstar
pointspwl += mean_pointspwl
pointsrw += mean_pointsrw

posstar.set_points(pointsstar)
pospwl.set_points(pointspwl)
posrw.set_points(pointsrw)

ax_star_traj.set_position(posstar)
ax_pwl_traj.set_position(pospwl)
ax_rw_traj.set_position(posrw)

ax_pos(ax, 0, 0, 1., 1.)
ax_pos(ax2, -0.02, -0.01, 0.65, 0.65)
ax_pos(ax2b, -0.12, -0.01, 0.65, 0.65)

os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(plot_fname, dpi=300)
