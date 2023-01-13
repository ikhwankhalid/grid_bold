import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from regex import W
from functions.gridfcts import (
    meanfr_repsupp
)
from utils.utils import get_hexsym
import utils.settings as settings
from joblib import Parallel, delayed
from tqdm import tqdm


taus = settings.taus
ws = settings.ws
N = settings.N
phbins = settings.phbins
bins = 500
rep = 40

# gr60s = np.zeros((len(taus), len(ws)))
# grpaths = np.zeros((len(taus), len(ws)))
# # gr2s = np.zeros((len(taus), len(ws), phbins, bins))
# meanfrs = np.zeros((len(taus), len(ws), phbins))


nxticks = 4
xtickstep = int(len(taus) / nxticks)
ytickstep = 5


gr60s_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"gr60s.pkl"
)
grpaths_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"grpaths.pkl"
)
# gr2s_fname = os.path.join(
#     settings.loc,
#     "repsupp",
#     "fig5",
#     "meanfr",
#     f"gr2s.pkl"
# )
mfrs_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"mfrs.pkl"
)
offs_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"offs.pkl"
)
plotnew_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"gr60s_new.png"
)
plotgr2s_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"gr2s_new.png"
)
plotmfrs_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"mfrs_new.png"
)
offsplot_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    f"offs.png"
)


def mfunc(i):
    gr60s = np.zeros((len(taus), len(ws)))
    grpaths = np.zeros((len(taus), len(ws)))
    meanfrs = np.zeros((len(taus), len(ws), phbins))
    for it, tau in enumerate(taus):
        for iw, w in enumerate(ws):
            plot = False
            # tau = 3.6, w = 1
            # if it == 10 and iw == 6:
            #     plot = True
            # else:
            #     plot = False
            gr60s[it, iw], grpaths[it, iw], meanfrs[it, iw, :], offs, _ = meanfr_repsupp(
                rmax = 300,
                bins = bins,
                phbins=phbins,
                N=N,
                tau_rep=tau,
                w_rep=w,
                plot=plot
            )
        print(f"Done with {it} tau")

    return gr60s


if not os.path.exists(gr60s_fname):
    # for it, tau in enumerate(taus):
    #     for iw, w in enumerate(ws):
    #         plot = False
    #         # tau = 3.6, w = 1
    #         # if it == 10 and iw == 6:
    #         #     plot = True
    #         # else:
    #         #     plot = False
    #         gr60s[it, iw], grpaths[it, iw], meanfrs[it, iw, :], offs, _ = meanfr_repsupp(
    #             rmax = 300,
    #             bins = bins,
    #             phbins=phbins,
    #             N=N,
    #             tau_rep=tau,
    #             w_rep=w,
    #             plot=plot
    #         )
    #     print(f"Done with {it} tau")
    gr60s = Parallel(
        n_jobs=-1, verbose=100)(delayed(mfunc)(i) for i in tqdm(range(rep))
    )
    gr60s = np.moveaxis(np.array(gr60s), 0, -1)
    print(gr60s.shape)
    os.makedirs(os.path.dirname(gr60s_fname), exist_ok=True)
    with open(gr60s_fname, "wb") as f:
        pickle.dump(gr60s, f)
    # with open(grpaths_fname, "wb") as f:
    #     pickle.dump(grpaths, f)
    # with open(gr2s_fname, "wb") as f:
    #     pickle.dump(gr2s, f)
    # with open(mfrs_fname, "wb") as f:
    #     pickle.dump(meanfrs, f)
    # with open(offs_fname, "wb") as f:
    #     pickle.dump(offs, f)
else:
    gr60s = pickle.load(open(gr60s_fname, "rb"))
    # gr2s = pickle.load(open(gr2s_fname, "rb"))
    # meanfrs = pickle.load(open(mfrs_fname, "rb"))
    # offs = pickle.load(open(offs_fname, "rb"))


plt.figure(figsize=(15, 8))
plt.rcParams.update({'font.size': int(settings.fs * 1.2)})
plt.imshow(np.median(gr60s, axis=-1).T, cmap="viridis", origin="lower", aspect="auto")
# plt.xticks(np.arange(1, len(taus) - 1, step=xtickstep), taus[1::xtickstep])
# plt.yticks(np.arange(len(ws), step=ytickstep), ws[::ytickstep])
plt.xlabel("$\\tau_r$", fontsize=int(settings.fs*2))
plt.ylabel("$w_r$", fontsize=int(settings.fs*2))
plt.title("parameter search (star-like run)", y=1.00, pad=14)
# plt.title("With new hexasymmetry measure")
cbar = plt.colorbar(fraction=0.020, pad=0.04)
cbar.set_label("hexasymmetry", fontsize=int(settings.fs*1.5))
plt.tight_layout()
plt.savefig(plotnew_fname)

# plt.figure(figsize=(10, 8))
# plt.plot(np.sum(gr2s[5, -1, :, :], axis=1))
# plt.savefig(plotgr2s_fname)

# plt.figure(figsize=(10, 8))
# plt.plot(meanfrs[5, -1, :])
# plt.savefig(plotmfrs_fname)

# plot offset figure
# plt.figure(figsize=(10, 8))
# plt.scatter(offs[0], offs[1])
# plt.savefig(offsplot_fname)