# %%
from functions.gridfcts import (
    gridpop_clustering,
    gridpop_conj,
    gridpop_repsupp,
    gen_offsets,
    get_offsets,
    traj
)
from utils.data_handler import load_data
from utils.utils import (
    convert_to_rhombus,
    get_hexsym,
    get_pathsym,
    get_plotting_grid
)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import utils.settings as settings
import time
from datetime import timedelta


def sim_bnds(
    traj_type: str,
    num_trajecs: int = settings.num_trajecs,
    n_torts : tuple = settings.n_torts,
    n_cells: int = 1024,
    grsc: float = settings.grsc,
    N: int = settings.N,
    phbins: int = settings.phbins,
    offs_idx: int = 50,
    hypothesis: str = None
):
    # check for valid hypothesis argument
    assert hypothesis in ["repsupp", "clustering", "conjunctive"], \
        "hypothesis must be 'repsupp', 'clustering', or 'conjunctive'"
    print("performing torts simulations for ", hypothesis)

    # get tortuosities
    torts = np.array([1e-3, 0.5])

    # output arrays initialisation
    fr_arr = np.zeros((num_trajecs, len(torts), n_cells))

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(settings.loc, hypothesis, "tort_mfr"),
        exist_ok=True
    )
    fr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "tort_mfr",
        f"{traj_type}",
        f"fr_torts.pkl"
    )

    # if firing rate array pickle files don't exist, run simulations
    if not os.path.exists(fr_fname):
        for i in range(num_trajecs):
            # simulate grid cell activity for each trajectory
            if hypothesis == "repsupp" or hypothesis == "conjunctive":
                ox, oy = gen_offsets(N=n_cells)
            elif hypothesis == "clustering":
                ox, oy = gen_offsets(N=n_cells, kappacl=settings.kappa_si)
            oxr, oyr = convert_to_rhombus(ox, oy)
            for j, tort in enumerate(torts):
                trajec = traj(
                    dt=settings.dt,
                    tmax=settings.tmax,
                    sp=settings.speed,
                    init_dir=settings.init_dir,
                    dphi=tort,
                    sq_bound=False
                )
                if hypothesis == "clustering":
                    _, mean_fr, fr, summed_fr = gridpop_clustering(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )
                elif hypothesis == "conjunctive":
                    _, mean_fr, fr, summed_fr = gridpop_conj(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr,
                        propconj=settings.propconj_i,
                        kappa=settings.kappac_i,
                        jitter=settings.jitterc_i
                    )
                else:
                    _, mean_fr, fr, summed_fr = gridpop_repsupp(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )

                fr_arr[i, j, :] = np.mean(fr, axis=-1)

            os.makedirs(os.path.dirname(fr_fname), exist_ok=True)
            with open(fr_fname, "wb") as f:
                pickle.dump(fr_arr, f)
    else:
        print("torts simulations already exist")

    print("Generating plots for simulated torts data...")

    # load arrays for plotting
    # slices = np.random.randint(0, 100)
    indices = np.random.choice(np.arange(0, 100), 5, replace=False)
    fr_arr = np.load(fr_fname, allow_pickle=True)
    lowtortfr = fr_arr[indices, 0, :].flatten()
    hitortfr = fr_arr[indices, 1, :].flatten()
    print("low: ", np.mean(lowtortfr), "high: ", np.mean(hitortfr))

    # calculate path-normalised hexasymmetry:
    # trajectories with path hexasymmetries an order of magnitude smaller than 
    # the mean are discarded as outliers
    hexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "tort_mfr",
        f"{traj_type}",
        f"hist.png"
    )

    fig = plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': settings.fs})
    spec = fig.add_gridspec(
        ncols=1, 
        nrows=1,
    )

    # plot hexsyms
    ax1 = fig.add_subplot(spec[0, 0])
    plt.rcParams.update({'font.size': settings.fs})
    plt.hist(lowtortfr, bins=100, alpha=0.5, label=r"$\sigma_\theta = 0.001$", range=(0.7, 0.9))
    plt.hist(hitortfr, bins=100, alpha=0.5, label=r"$\sigma_\theta = 0.5$", range=(0.7, 0.9))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title("Distributions of grid cell firing rates (N=1024)\nfor a random walk with two different tortuosity values.\n5 realisations")
    plt.xlabel("Firing rate (spk/s)")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    plt.savefig(hexplotloc)


if __name__ == "__main__":
    start_time = time.monotonic()
    for traj_type in ["rw"]:
        # for hypothesis in ["conjunctive", "clustering", "repsupp"]:
        for hypothesis in ["repsupp"]:
            sim_bnds(
                traj_type=traj_type,
                num_trajecs=100,
                hypothesis=hypothesis
            )
    end_time = time.monotonic()
    print(
        "torts simulations finished in: ", 
        timedelta(seconds=end_time - start_time)
    )
