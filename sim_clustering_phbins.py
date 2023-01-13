# %%
from functions.gridfcts import (
    gridpop_clustering,
    gridpop_conj,
    gridpop_repsupp,
    gen_offsets,
    get_offsets,
    traj_pwl
)
from utils.data_handler import load_data
from utils.utils import (
    convert_to_rhombus,
    get_hexsym,
    get_pathsym
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
    ntrials: int = settings.ntrials_pwl_phbins,
    n_phbins : tuple = settings.n_pwl_phbins,
    phbins_range: tuple = settings.pwl_phbins_range,
    grsc: float = settings.grsc,
    N: int = settings.N,
    phbins: int = settings.phbins,
    offs_idx: int = 50,
    hypothesis: str = None
):
    # check for valid hypothesis argument
    assert hypothesis in ["repsupp", "clustering", "conjunctive"], \
        "hypothesis must be 'repsupp', 'clustering', or 'conjunctive'"
    print("performing lengths simulations for ", hypothesis)

    # get tortuosities
    phbins_list = np.logspace(phbins_range[0], phbins_range[1], n_phbins, endpoint=True, base=2).astype(int)

    # point to directory of trajectory type
    if traj_type == "rw":
        traj_path = settings.rw_loc
    elif traj_type == "pwl":
        traj_path = settings.pwl_loc

    # output arrays initialisation
    fr_arr = np.zeros((ntrials, n_phbins, phbins))
    hexes = np.zeros((ntrials, n_phbins))
    pathhexes = np.zeros((ntrials, n_phbins))

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(settings.loc, hypothesis, "pwl_lengths"),
        exist_ok=True
    )
    fr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "pwl_phbins",
        f"{traj_type}",
        f"fr_lengths.pkl"
    )
    hex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "pwl_phbins",
        f"{traj_type}",
        f"hex_lengths.pkl"
    )
    pathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "pwl_phbins",
        f"{traj_type}",
        f"pathhex_lengths.pkl"
    )

    # if firing rate array pickle files don't exist, run simulations
    if not os.path.exists(hex_fname):
        # simulate grid cell activity for each trajectory
        for i in range(ntrials):
            if hypothesis == "repsupp" or hypothesis == "conjunctive":
                ox, oy = gen_offsets()
            elif hypothesis == "clustering":
                ox, oy = gen_offsets(kappacl=settings.kappa_si)
            oxr, oyr = convert_to_rhombus(ox, oy)
            if i >= ntrials:
                break
            if i == 1:
                start_time = time.monotonic()
            for j, phbins in enumerate(phbins_list):
                if j == 1:
                    start_time2 = time.monotonic()
                trajec = traj_pwl(
                    phbins,
                    settings.rmax,
                    settings.dt,
                    sp=settings.speed
                )
                if hypothesis == "clustering":
                    _, mean_fr, _, summed_fr = gridpop_clustering(
                        N,
                        grsc,
                        settings.phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )
                elif hypothesis == "conjunctive":
                    _, mean_fr, _, summed_fr = gridpop_conj(
                        N,
                        grsc,
                        settings.phbins,
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
                        settings.phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )

                fr_arr[i, j, :] = mean_fr
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
                        "One length simulation finished in: ", 
                        timedelta(seconds=end_time2 - start_time2)
                    )
                    print(
                        f"{n_phbins} simulations estimated in: ", 
                        timedelta(seconds=(end_time2 - start_time2) * n_phbins)
                    )

            if i == 1:
                end_time = time.monotonic()
                print(
                    "One set of trajectory simulations finished in: ", 
                    timedelta(seconds=end_time - start_time)
                )
                print(
                    f"{ntrials} simulations estimated in: ", 
                    timedelta(seconds=(end_time - start_time) * ntrials)
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
        print("lengths simulations already exist")

    print("Generating plots for simulated lengths data...")

    # load arrays for plotting
    fr_arr = np.load(fr_fname, allow_pickle=True)
    hexes = np.load(hex_fname, allow_pickle=True)
    pathhexes = np.load(pathhex_fname, allow_pickle=True)

    # calculate path-normalised hexasymmetry:
    # trajectories with path hexasymmetries an order of magnitude smaller than 
    # the mean are discarded as outliers
    # hexes_normed = hexes / pathhexes
    # hexes_normed = np.delete(
    #     hexes / pathhexes,
    #     np.where(pathhexes < 0.001)[0],
    #     axis=0
    # )

    # plot hexsyms
    hexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "pwl_phbins",
        f"{traj_type}",
        f"hex.png"
    )
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': settings.fs})
    plt.plot(
        phbins_list[4:],
        np.mean(hexes[:, 4:], axis=0),
        color="black",
        lw=2,
        label=r"Mean $|\tilde{A}_6|$"
    )
    plt.fill_between(
        phbins_list[4:],
        np.mean(hexes[:, 4:], axis=0) -
        np.std(hexes[:, 4:], axis=0) / np.sqrt(hexes.shape[0]),
        np.mean(hexes[:, 4:], axis=0) +
        np.std(hexes[:, 4:], axis=0) / np.sqrt(hexes.shape[0]),
        alpha=0.15,
        color="black"
    )
    # plt.plot(
    #     phbins_list[4:],
    #     np.mean(pathhexes[:, 4:] * np.mean(fr_arr, axis=(0, -1))[4:], axis=0),
    #     lw=2,
    #     label=r"Mean $|\tilde{T}_6| \cdot \tilde{A}_0$"
    # )
    # plt.fill_between(
    #     phbins_list[4:],
    #     np.mean(pathhexes[:, 4:] * np.mean(fr_arr, axis=(0, -1))[4:], axis=0) -
    #     np.std(pathhexes[:, 4:] * np.mean(fr_arr, axis=(0, -1))[4:], axis=0) / np.sqrt(pathhexes.shape[0]),
    #     np.mean(pathhexes[:, 4:] * np.mean(fr_arr, axis=(0, -1))[4:], axis=0) +
    #     np.std(pathhexes[:, 4:] * np.mean(fr_arr, axis=(0, -1))[4:], axis=0) / np.sqrt(pathhexes.shape[0]),
    #     alpha=0.2,
    # )
    # plt.plot(
    #     phbins_list[4:],
    #     1/np.sqrt(phbins_list[4:] * 300)+40,
    #     label=r"$c~\ /\sqrt{M}}$",
    #     zorder=0,
    #     linewidth=4
    # )
    plt.plot(
        phbins_list[4:],
        1/np.sqrt(phbins_list[4:] * 300)*4000,
        label=r"$c~\ /\sqrt{M}}$",
        zorder=0,
        linestyle="dashdot",
        linewidth=2,
        color="black"
    )
    plt.margins(0.01, 0.15)
    # plt.title(f"Structure-function mapping, piecewise linear walk\nmean over {ntrials} trials")
    plt.xlabel("Number of angles", labelpad=20)
    plt.ylabel("Hexasymmetry (spk/s)")
    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.vlines(360, 1e0 - 2 * 1e-1, 1e2, ls="--", color="black", clip_on=False)
    plt.ylim(1e0, 1e2)
    plt.text(0.45, 0.17, "360", fontsize=settings.fs * 0.8, transform=plt.gcf().transFigure)
    ax.set_aspect('equal', adjustable=None)
    plt.legend(prop={'size': 0.7 * settings.fs})
    plt.tight_layout()
    os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    plt.savefig(hexplotloc)
    plt.close()

    # # plot normalised hexsyms
    # normedhexplotloc = os.path.join(
    #     settings.loc,
    #     "plots",
    #     hypothesis,
    #     "pwl_phbins",
    #     f"{traj_type}",
    #     f"normedhex.png"
    # )
    # plt.figure(figsize=(12, 4))
    # plt.rcParams.update({'font.size': settings.fs})
    # plt.plot(
    #     phbins_list,
    #     np.median(hexes_normed, axis=0),
    #     marker="^",
    #     lw=2,
    #     markersize=8
    # )
    # plt.fill_between(
    #     phbins_list,
    #     np.median(hexes_normed, axis=0) -
    #     np.std(hexes_normed, axis=0) / np.sqrt(hexes_normed.shape[0]),
    #     np.median(hexes_normed, axis=0) +
    #     np.std(hexes_normed, axis=0) / np.sqrt(hexes_normed.shape[0]),
    #     alpha=0.2,
    # )
    # plt.margins(0.01, 0.15)
    # plt.title(hypothesis+f", median over {ntrials} trials")
    # plt.xlabel("Path segment length (cm)")
    # plt.ylabel("Hexasymmetry (norm)\n(Spks/s)")
    # # plt.yscale("log")
    # plt.tight_layout()
    # os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    # plt.savefig(normedhexplotloc)
    # plt.close()

    # get path symmetry plot
    pathhexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "pwl_phbins",
        f"{traj_type}",
        f"pathhex.png"
    )
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': 18})
    plt.plot(
        phbins_list[4:],
        np.median(pathhexes[:, 4:], axis=0),
        marker="^",
        lw=2,
        markersize=8,
        color="darkorange"
    )
    plt.fill_between(
        phbins_list[4:],
        np.median(pathhexes[:, 4:], axis=0) -
        np.std(pathhexes[:, 4:], axis=0) / np.sqrt(pathhexes.shape[0]),
        np.median(pathhexes[:, 4:], axis=0) +
        np.std(pathhexes[:, 4:], axis=0) / np.sqrt(pathhexes.shape[0]),
        alpha=0.2,
        color="darkorange"
    )
    plt.margins(0.01, 0.15)
    plt.xlabel("Number of angles")
    plt.ylabel("path hexasymmetry (a.u.)")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(5e-15, 2e-13)
    plt.tight_layout()

    os.makedirs(os.path.dirname(pathhexplotloc), exist_ok=True)
    plt.savefig(pathhexplotloc)
    plt.close()


if __name__ == "__main__":
    start_time = time.monotonic()
    for traj_type in ["rw"]:
        # for hypothesis in ["conjunctive", "clustering", "repsupp"]:
        for hypothesis in ["clustering"]:
            sim_bnds(
                traj_type=traj_type,
                ntrials=settings.ntrials_pwl_phbins,
                hypothesis=hypothesis
            )
    end_time = time.monotonic()
    print(
        "lengths simulations finished in: ", 
        timedelta(seconds=end_time - start_time)
    )


# %%
