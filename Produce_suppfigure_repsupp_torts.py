"""
For historical reasons, this script generates two separate plots which were
combined in inkscape.
"""

from utils.grid_funcs import (
    gridpop_clustering,
    gridpop_conj,
    gridpop_repsupp,
    gen_offsets,
    traj,
    traj_pwl
)
from utils.utils import (
    convert_to_rhombus,
    get_hexsym,
    get_pathsym,
    get_plotting_grid,
    get_t6upperbnd,
)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import settings
import time
from datetime import timedelta


plotloc = os.path.join(
    settings.loc,
    "plots",
    "trajectories",
    "rw",
    "torts.png"
)


def sim_bnds(
    traj_type: str,
    num_trajecs: int = settings.num_trajecs,
    n_torts: tuple = settings.n_torts,
    tort_range: tuple = settings.tort_range,
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
    torts = np.logspace(
        np.log10(tort_range[0]),
        np.log10(tort_range[1]),
        n_torts,
        endpoint=True
    )

    # output arrays initialisation
    fr_arr = np.zeros((num_trajecs, n_torts))
    hexes = np.zeros((num_trajecs, n_torts))
    pathhexes = np.zeros((num_trajecs, n_torts))

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(settings.loc, hypothesis, "torts"),
        exist_ok=True
    )
    fr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "torts",
        f"{traj_type}",
        "fr_torts.pkl"
    )
    hex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "torts",
        f"{traj_type}",
        "hex_torts.pkl"
    )
    pathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "torts",
        f"{traj_type}",
        "pathhex_torts.pkl"
    )

    # if firing rate array pickle files don't exist, run simulations
    if not os.path.exists(hex_fname):
        # simulate grid cell activity for each trajectory
        for i in range(num_trajecs):
            if hypothesis == "repsupp" or hypothesis == "conjunctive":
                ox, oy = gen_offsets()
            elif hypothesis == "clustering":
                ox, oy = gen_offsets(kappacl=settings.kappa_si)
            oxr, oyr = convert_to_rhombus(ox, oy)
            if i >= num_trajecs:
                break
            if i == 1:
                start_time = time.monotonic()
            for j, tort in enumerate(torts):
                if j == 1:
                    start_time2 = time.monotonic()
                trajec = traj(
                    dt=settings.dt,
                    tmax=settings.tmax,
                    sp=settings.speed,
                    init_dir=settings.init_dir,
                    dphi=tort,
                    sq_bound=False
                )
                if hypothesis == "clustering":
                    _, mean_fr, _, summed_fr = gridpop_clustering(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )
                elif hypothesis == "conjunctive":
                    _, mean_fr, _, summed_fr = gridpop_conj(
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
                    _, mean_fr, _, summed_fr = gridpop_repsupp(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )

                fr_arr[i, j] = np.mean(summed_fr)
                hexes[i, j] = get_hexsym(
                    summed_fr,
                    trajec
                )
                pathhexes[i, j] = get_pathsym(
                    trajec
                )

                if j == 1:
                    end_time2 = time.monotonic()
                    print(
                        "One size simulation finished in: ",
                        timedelta(seconds=end_time2 - start_time2)
                    )
                    print(
                        f"{n_torts} simulations estimated in: ",
                        timedelta(seconds=(end_time2 - start_time2) * n_torts)
                    )

            if i == 1:
                end_time = time.monotonic()
                print(
                    "One set of trajectory simulations finished in: ",
                    timedelta(seconds=end_time - start_time)
                )
                print(
                    f"{num_trajecs} simulations estimated in: ",
                    timedelta(seconds=(end_time - start_time) * num_trajecs)
                )

        os.makedirs(os.path.dirname(fr_fname), exist_ok=True)
        os.makedirs(os.path.dirname(hex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(pathhex_fname), exist_ok=True)
        with open(fr_fname, "wb") as f:
            pickle.dump(fr_arr, f)
        with open(hex_fname, "wb") as f:
            pickle.dump(hexes, f)
        with open(pathhex_fname, "wb") as f:
            pickle.dump(pathhexes, f)
    else:
        print("torts simulations already exist")

    print("Generating plots for simulated torts data...")

    # load arrays for plotting
    fr_arr = np.load(fr_fname, allow_pickle=True)
    hexes = np.load(hex_fname, allow_pickle=True)
    pathhexes = np.load(pathhex_fname, allow_pickle=True)

    hexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "torts",
        f"{traj_type}",
        "hex.png"
    )

    fig = plt.figure(figsize=(12, 12))
    plt.rcParams.update({'font.size': settings.fs})
    spec = fig.add_gridspec(
        ncols=1,
        nrows=4,
        height_ratios=[1, 0.75, 0.75, 1]
    )

    # plot hexsym, path hexsym
    ax1 = fig.add_subplot(spec[0, 0])
    plt.rcParams.update({'font.size': settings.fs})
    plt.plot(
        torts,
        np.mean(hexes, axis=0),
        lw=2,
        label=r"mean $|\tilde{A}_6|$"
    )
    plt.fill_between(
        torts,
        np.mean(hexes, axis=0) -
        np.std(hexes, axis=0) / np.sqrt(hexes.shape[0]),
        np.mean(hexes, axis=0) +
        np.std(hexes, axis=0) / np.sqrt(hexes.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        torts,
        np.mean(pathhexes * np.mean(fr_arr, axis=0), axis=0),
        lw=2,
        label=r"mean $|\tilde{T}_6| \cdot \tilde{A}_0$"
    )
    plt.fill_between(
        torts,
        np.mean(pathhexes * np.mean(fr_arr, axis=0), axis=0) -
        np.std(pathhexes * np.mean(fr_arr, axis=0), axis=0) /
        np.sqrt(pathhexes.shape[0]),
        np.mean(pathhexes * np.mean(fr_arr, axis=0), axis=0) +
        np.std(pathhexes * np.mean(fr_arr, axis=0), axis=0) /
        np.sqrt(pathhexes.shape[0]),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.suptitle("Repetition-suppression")
    plt.ylabel("Hexasymmetry\n(spk/s)")
    plt.locator_params(axis='y', nbins=3)
    plt.ylim(0, 40)
    plt.xticks(np.linspace(0.1, 0.7, 7, endpoint=True), [])
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot difference in hexsyms
    hexdiff = hexes - pathhexes * np.mean(fr_arr, axis=0)
    ax2 = fig.add_subplot(spec[1, 0])
    plt.rcParams.update({'font.size': settings.fs})
    plt.plot(
        torts,
        np.mean(hexdiff, axis=0),
        lw=2,
        color="black"
    )
    plt.fill_between(
        torts,
        np.mean(hexdiff, axis=0) -
        np.std(hexdiff, axis=0) / np.sqrt(hexdiff.shape[0]),
        np.mean(hexdiff, axis=0) +
        np.std(hexdiff, axis=0) / np.sqrt(hexdiff.shape[0]),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.ylabel("Mean difference\n(spk/s)")
    plt.locator_params(axis='y', nbins=3)
    plt.ylim(-10, 40)
    plt.yticks([0, 20, 40])
    plt.xticks(np.linspace(0.1, 0.7, 7, endpoint=True), [])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # mean fr
    ax3 = fig.add_subplot(spec[2, 0])
    plt.rcParams.update({'font.size': settings.fs})
    plt.plot(
        torts,
        np.mean(fr_arr, axis=0),
        # marker="^",
        lw=2,
        color="black",
        # markersize=8
    )
    plt.fill_between(
        torts,
        np.mean(fr_arr, axis=0) -
        np.std(fr_arr, axis=(0, -1)) / np.sqrt(fr_arr.shape[0]),
        np.mean(fr_arr, axis=0) +
        np.std(fr_arr, axis=(0, -1)) / np.sqrt(fr_arr.shape[0]),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.locator_params(axis='y', nbins=3)
    plt.ylim(820, 860)
    plt.ylabel("Population\nfiring rate\n(spk/s)")
    plt.xticks(np.linspace(0.1, 0.7, 7, endpoint=True), [])
    plt.yticks([820, 840, 860])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot pathsym
    ax4 = fig.add_subplot(spec[3, 0])
    plt.rcParams.update({'font.size': 18})
    plt.plot(
        torts,
        np.mean(pathhexes, axis=0),
        # marker="^",
        lw=2,
        # markersize=8,
        color="black",
        label=r"mean $|\tilde{T}_6|$"
    )
    plt.plot(
        torts,
        get_t6upperbnd(torts, settings.dt, int(settings.tmax / settings.dt)),
        ls="--",
        lw=2,
        # markersize=8,
        color="black",
        label=r"mean $|\tilde{T}_6|$"
    )
    plt.margins(0.01, 0.05)
    plt.xlabel(r"Movement tortuosity $\sigma_\theta~(rad/s^{1/2}$)")
    plt.ylabel("Path hexasymmetry")
    plt.locator_params(axis='y', nbins=3)
    plt.ylim(0, 0.04)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    plt.savefig(hexplotloc, dpi=300)


if __name__ == "__main__":
    start_time = time.monotonic()
    for traj_type in ["rw"]:
        for hypothesis in ["repsupp"]:
            sim_bnds(
                traj_type=traj_type,
                num_trajecs=settings.ntrials_tort,
                hypothesis=hypothesis
            )
    end_time = time.monotonic()
    print(
        "torts simulations finished in: ",
        timedelta(seconds=end_time - start_time)
    )

    fig = plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': settings.fs})
    offs = [0, 0]
    spec = fig.add_gridspec(
        ncols=4,
        nrows=1,
    )

    init_dirs = np.linspace(0, 2*np.pi, 5, endpoint=False)
    colours = ["red", "yellow", "lime", "hotpink", "lightgray"]

    ax1 = fig.add_subplot(spec[0, 0])
    for i in range(len(init_dirs)):
        get_plotting_grid(
            traj_pwl(
                phbins=settings.phbins,
                rmax=settings.rmax,
                dt=settings.dt
            ),
            offs,
            extent=600,
            title=r"$p-l$",
            titlepos=[0.5, 1.05],
            yticks=[-600, 0, 600],
            xticks=[0, 600],
            trajcolor=colours[i],
            trajprop=0.015,
            ylabel="y (cm)",
            xlabel="x (cm)"
        )

    ax2 = fig.add_subplot(spec[0, 1])
    for i in range(len(init_dirs)):
        trajec = traj(
            dt=settings.dt,
            tmax=60,
            sp=settings.speed,
            init_dir=init_dirs[i],
            dphi=0.1,
            sq_bound=False
        )
        get_plotting_grid(
            trajec,
            offs,
            extent=600,
            title="rand\n" + r"$\sigma_\theta=0.1~rad/s^{1/2}$",
            titlepos=[0.5, 1.05],
            xticks=[-600, 0, 600],
            trajcolor=colours[i],
            xlabel="x (cm)"
        )

    ax3 = fig.add_subplot(spec[0, 2])
    for i in range(len(init_dirs)):
        trajec = traj(
            dt=settings.dt,
            tmax=60,
            sp=settings.speed,
            init_dir=init_dirs[i],
            dphi=0.25,
            sq_bound=False
        )
        get_plotting_grid(
            trajec,
            offs,
            extent=600,
            title="rand\n" + r"$\sigma_\theta=0.25~rad/s^{1/2}$",
            titlepos=[0.5, 1.05],
            xticks=[-600, 0, 600], trajcolor=colours[i],
            xlabel="x (cm)"
        )

    ax3 = fig.add_subplot(spec[0, 3])
    for i in range(len(init_dirs)):
        trajec = traj(
            dt=settings.dt,
            tmax=60,
            sp=settings.speed,
            init_dir=init_dirs[i],
            dphi=0.5,
            sq_bound=False
        )
        get_plotting_grid(
            trajec,
            offs,
            extent=600,
            title="rand\n" + r"$\sigma_\theta=0.5~rad/s^{1/2}$",
            titlepos=[0.5, 1.05],
            xticks=[-600, 0, 600], trajcolor=colours[i],
            xlabel="x (cm)"
        )

    plotloc = os.path.join(
        settings.loc,
        "plots",
        "trajectories",
        "rw",
        "torts.png"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(plotloc), exist_ok=True)
    plt.savefig(plotloc, dpi=300)
    plt.close()
