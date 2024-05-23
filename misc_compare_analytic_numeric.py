"""
This script creates a plot which compares the heading dependent firing rates
when numerically simulating equation 8 versus using the analytical derivation
in equation 32 of the manuscript. The plot was not ued in the final version
of the manuscript.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import settings
import os
import pickle
matplotlib.use('Agg')

###############################################################################
# Parameters                                                                  #
###############################################################################
fs = settings.fs                        # default font size for plots
lw = 2.                                 # line width for line plots

###############################################################################
# Plotting                                                                    #
###############################################################################
fig = plt.figure(figsize=(20, 14))
spec = fig.add_gridspec(
    ncols=3,
    nrows=5,
    height_ratios=[0.5, 1, 1, 0.5, 1],
    width_ratios=[1, 1, 1]
)

###############################################################################
# Conjunctive hypothesis                                                      #
###############################################################################
# star-like walk
conjstar_nfname = os.path.join(
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


with open(conjstar_nfname, 'rb') as f:
    nmfr_conjstar, ngr60_conjstar = pickle.load(f)
with open(conjstar_afname, 'rb') as f:
    amfr_conjstar = np.load(f)
os.makedirs(os.path.dirname(os.path.join(
        settings.loc,
        "compare_analytic_numeric",
        "conjstar.png"
        )
    ),
    exist_ok=True
)


ax_conjstar = fig.add_subplot(spec[1, 0])
ax_conjstar.spines['top'].set_visible(False)
ax_conjstar.spines['right'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_conjstar,
    ls="--",
    lw=lw,
    zorder=10,
    label="Numerical"
)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_conjstar,
    lw=lw,
    label="Analytical"
)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [], fontsize=fs)
plt.yticks([0, 5000], fontsize=fs)
plt.ylabel('Conjunctive\nTotal firing \nrate (spk/s)', fontsize=fs)
plt.title("Star-like run", fontsize=fs)
plt.ylim([0, 5100])


# pwl walk
conjpwl_nfname = os.path.join(
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
            """
            conjunctive_pwl_N1024_nphi360_rmin0_rmax3_sigma0_kappa50_ratio1
            .npy
            """
        )
)


with open(conjpwl_nfname, 'rb') as f:
    nmfr_conjpwl, ngr60_conjpwl = pickle.load(f)
with open(conjpwl_afname, 'rb') as f:
    amfr_conjpwl = np.load(f)


ax_conjpwl = fig.add_subplot(spec[1, 1])
ax_conjpwl.spines['top'].set_visible(False)
ax_conjpwl.spines['right'].set_visible(False)
ax_conjpwl.spines['left'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_conjpwl,
    ls="--",
    lw=lw,
    zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_conjpwl, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [], fontsize=fs)
plt.yticks([], fontsize=fs)
plt.title("PWL run", fontsize=fs)
plt.ylim([0, 5100])


# random walk
conjrw_nfname = os.path.join(
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
            """
            conjunctive_rw_N1024_nphi360_T9000_dt0_01_v0_1_sigmatheta0_5_sigma0
            _kappa50_ratio1.npy
            """
        )
)


with open(conjrw_nfname, 'rb') as f:
    nmfr_conjrw, ngr60_conjrw = pickle.load(f)
with open(conjrw_afname, 'rb') as f:
    amfr_conjrw = np.load(f)


ax_conjrw = fig.add_subplot(spec[1, 2])
ax_conjrw.spines['top'].set_visible(False)
ax_conjrw.spines['right'].set_visible(False)
ax_conjrw.spines['left'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_conjrw,
    ls="--",
    lw=lw,
    zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_conjrw, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [], fontsize=fs)
plt.yticks([], fontsize=fs)
plt.title("random walk", fontsize=fs)
plt.ylim([0, 5100])
# plt.legend()


###############################################################################
# Clustering hypothesis                                                       #
###############################################################################
# star-like walk
cluststar_nfname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "star_fr.pkl"
)
cluststar_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "conjunctive_sw_N1024_nphi360_rmin0_rmax3.npy"
        )
)


with open(cluststar_nfname, 'rb') as f:
    nmfr_cluststar = pickle.load(f)
with open(cluststar_afname, 'rb') as f:
    amfr_cluststar = np.load(f)


ax_cluststar = fig.add_subplot(spec[2, 0])
ax_cluststar.spines['top'].set_visible(False)
ax_cluststar.spines['right'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_cluststar,
    ls="--",
    lw=lw,
    zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_cluststar, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks([0, 3000], fontsize=fs)
plt.xlabel('Movement direction (deg)', fontsize=fs)
plt.ylabel('Clustering\nTotal firing \nrate (spk/s)', fontsize=fs)
plt.ylim([0, 3100])


# pwl-like walk
clustpwl_nfname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "pwl_fr.pkl"
)
clustpwl_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "clustering_pwl_N1024_nphi360_rmin0_rmax3_kappa10.npy"
        )
)


with open(clustpwl_nfname, 'rb') as f:
    nmfr_clustpwl = pickle.load(f)
with open(clustpwl_afname, 'rb') as f:
    amfr_clustpwl = np.load(f)


ax_clustpwl = fig.add_subplot(spec[2, 1])
ax_clustpwl.spines['top'].set_visible(False)
ax_clustpwl.spines['right'].set_visible(False)
ax_clustpwl.spines['left'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_clustpwl,
    ls="--",
    lw=lw,
    zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_clustpwl, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks([], fontsize=fs)
plt.xlabel('Movement direction (deg)', fontsize=fs)
plt.ylim([0, 3100])


# random walk
clustrw_nfname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "rw_fr.pkl"
)
clustrw_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            """
            clustering_rw_N1024_nphi360_T9000_dt0_01_v0_1_sigmatheta0_5_kappa10
            .npy
            """
        )
)


with open(clustrw_nfname, 'rb') as f:
    nmfr_clustrw = pickle.load(f)
with open(clustrw_afname, 'rb') as f:
    amfr_clustrw = np.load(f)


ax_clustrw = fig.add_subplot(spec[2, 2])
ax_clustrw.spines['top'].set_visible(False)
ax_clustrw.spines['right'].set_visible(False)
ax_clustrw.spines['left'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_clustrw,
    ls="--",
    lw=lw,
    zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_clustrw, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks([], fontsize=fs)
plt.xlabel('Movement direction (deg)', fontsize=fs)
plt.ylim([0, 3100])


###############################################################################
# Conjunctive parameter variation                                             #
###############################################################################
# kappa_h=50, sigma_h=0
conjk50j0_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata1.pkl"
)
conjk50j0_afname = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            """
            conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma0_kappa50_ratio1.npy
            """
        )
)


with open(conjk50j0_nfname, 'rb') as f:
    nmfr_conjk50j0, ngr60_conjk50j0 = pickle.load(f)
with open(conjk50j0_afname, 'rb') as f:
    amfr_conjk50j0 = np.load(f)


ax_conjk50j0 = fig.add_subplot(spec[4, 0])
ax_conjk50j0.spines['top'].set_visible(False)
ax_conjk50j0.spines['right'].set_visible(False)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    nmfr_conjk50j0,
    ls="--",
    lw=lw,
    zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_conjk50j0, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks([0, 5000], fontsize=fs)
plt.xlabel('Movement direction (deg)', fontsize=fs)
plt.ylabel('Conjunctive\nTotal firing \nrate (spk/s)', fontsize=fs)
plt.title(r"$\kappa_h=50, \ \sigma_h=0$", fontsize=fs)
plt.ylim([0, 5100])


# kappa_h=25, sigma_h=5
conjk25j5_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata2.pkl"
)
conjk25j5_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            """
            conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma5_kappa25_ratio1.npy
            """
        )
)


with open(conjk25j5_nfname, 'rb') as f:
    nmfr_conjk25j5, ngr60_conjk25j5 = pickle.load(f)
with open(conjk25j5_afname, 'rb') as f:
    amfr_conjk25j5 = np.load(f)


ax_conjk25j5 = fig.add_subplot(spec[4, 1])
ax_conjk25j5.spines['top'].set_visible(False)
ax_conjk25j5.spines['right'].set_visible(False)
ax_conjk25j5.spines['left'].set_visible(False)
plt.plot(
        np.linspace(0, 360, settings.phbins),
        nmfr_conjk25j5,
        ls="--",
        lw=lw,
        zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_conjk25j5, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks([], fontsize=fs)
plt.xlabel('Movement direction (deg)', fontsize=fs)
plt.title(r"$\kappa_h=25, \ \sigma_h=5$", fontsize=fs)
plt.ylim([0, 5100])


# kappa_h=2.5, sigma_h=10
conjk2p5j10_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata3.pkl"
)
conjk2p5j10_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            """
            conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma10_kappa2_5_ratio1
            .npy
            """
        )
)


with open(conjk2p5j10_nfname, 'rb') as f:
    nmfr_conjk2p5j10, ngr60_conjk2p5j10 = pickle.load(f)
with open(conjk2p5j10_afname, 'rb') as f:
    amfr_conjk2p5j10 = np.load(f)


ax_conjk2p5j10 = fig.add_subplot(spec[4, 2])
ax_conjk2p5j10.spines['top'].set_visible(False)
ax_conjk2p5j10.spines['right'].set_visible(False)
ax_conjk2p5j10.spines['left'].set_visible(False)
plt.plot(
        np.linspace(0, 360, settings.phbins),
        nmfr_conjk2p5j10,
        ls="--",
        lw=lw,
        zorder=10
)
plt.plot(np.linspace(0, 360, settings.phbins), amfr_conjk2p5j10, lw=lw)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks([], fontsize=fs)
plt.xlabel('Movement direction (deg)', fontsize=fs)
plt.title(r"$\kappa_h=2.5, \ \sigma_h=10$", fontsize=fs)
plt.ylim([0, 5100])


###############################################################################
# Plot layout stuff                                                           #
###############################################################################
ax_text = fig.add_subplot(spec[0, 0:3])
plt.axis('off')
plt.text(
    0.5,
    0.5,
    r"Conjunctive and clustering hypotheses for each trajectory type",
    fontsize=fs*1.5,
    va='center',
    ha='center'
)


ax_text = fig.add_subplot(spec[3, 0:3])
plt.axis('off')
plt.text(
    0.5,
    0.5,
    r"Conjunctive hypothesis with varied $\kappa_h$ and $\sigma_h$",
    fontsize=fs*1.5,
    va='center',
    ha='center'
)


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, prop={'size': fs*1.2}, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(
    settings.loc,
    "compare_analytic_numeric",
    "compare.png"
    )
)
