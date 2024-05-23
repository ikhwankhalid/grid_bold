import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.grid_funcs import (
    meanfr_repsupp
)
import settings
from joblib import Parallel, delayed
from tqdm import tqdm

###############################################################################
# Parameters                                                                  #
###############################################################################
taus = settings.taus
ws = settings.ws
N = settings.N
phbins = settings.phbins
bins = 500
rep = 40

nxticks = 4
xtickstep = int(len(taus) / nxticks)
ytickstep = 5


gr60s_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "gr60s.pkl"
)
grpaths_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "grpaths.pkl"
)
mfrs_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "mfrs.pkl"
)
offs_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "offs.pkl"
)
plotnew_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "gr60s_new.png"
)
plotgr2s_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "gr2s_new.png"
)
plotmfrs_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "mfrs_new.png"
)
offsplot_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "offs.png"
)


def mfunc(i):
    gr60s = np.zeros((len(taus), len(ws)))
    grpaths = np.zeros((len(taus), len(ws)))
    meanfrs = np.zeros((len(taus), len(ws), phbins))
    for it, tau in enumerate(taus):
        for iw, w in enumerate(ws):
            plot = False
            (
                gr60s[it, iw],
                grpaths[it, iw],
                meanfrs[it, iw, :],
                offs,
                _
            ) = meanfr_repsupp(
                rmax=300,
                bins=bins,
                phbins=phbins,
                N=N,
                tau_rep=tau,
                w_rep=w,
                plot=plot
            )
        print(f"Done with {it} tau")

    return gr60s


if not os.path.exists(gr60s_fname):
    gr60s = Parallel(
        n_jobs=-1, verbose=100
    )(delayed(mfunc)(i) for i in tqdm(range(rep)))
    gr60s = np.moveaxis(np.array(gr60s), 0, -1)
    print(gr60s.shape)
    os.makedirs(os.path.dirname(gr60s_fname), exist_ok=True)
    with open(gr60s_fname, "wb") as f:
        pickle.dump(gr60s, f)
else:
    gr60s = pickle.load(open(gr60s_fname, "rb"))


plt.figure(figsize=(15, 8))
plt.rcParams.update({'font.size': int(settings.fs * 1.2)})
plt.imshow(
    np.median(gr60s, axis=-1).T, cmap="viridis", origin="lower", aspect="auto"
)
plt.xlabel("$\\tau_r$", fontsize=int(settings.fs*2))
plt.ylabel("$w_r$", fontsize=int(settings.fs*2))
plt.title("parameter search (star-like run)", y=1.00, pad=14)
cbar = plt.colorbar(fraction=0.020, pad=0.04)
cbar.set_label("hexasymmetry", fontsize=int(settings.fs*1.5))
plt.tight_layout()
plt.savefig(plotnew_fname)
