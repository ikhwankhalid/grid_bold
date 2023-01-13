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
    num_trajecs: int = settings.ntrials_finite,
    n_sizes : tuple = settings.n_sizes,
    size_range: tuple = settings.size_range,
    grsc: float = settings.grsc,
    N: int = settings.N,
    phbins: int = settings.phbins,
    offs_idx: int = 50,
    hypothesis: str = None
):
    # check for valid hypothesis argument
    assert hypothesis in ["repsupp", "clustering", "conjunctive"], \
        "hypothesis must be 'repsupp', 'clustering', or 'conjunctive'"
    print("performing sizes simulations for ", hypothesis)

    # append boundary sizes array with "infinite" boundaries
    sizes = np.append(
        np.linspace(size_range[0], size_range[1], n_sizes),
        float(1e6)
    )

    # point to directory of trajectory type
    if traj_type == "rw":
        traj_path = settings.rw_loc
    elif traj_type == "pwl":
        traj_path = settings.pwl_loc

    # output arrays initialisation
    circfr_arr = np.zeros((num_trajecs, n_sizes + 1, phbins))
    sqfr_arr = np.zeros((num_trajecs, n_sizes + 1, phbins))
    circhexes = np.zeros((num_trajecs, n_sizes + 1))
    sqhexes = np.zeros((num_trajecs, n_sizes + 1))
    circpathhexes = np.zeros((num_trajecs, n_sizes + 1))
    sqpathhexes = np.zeros((num_trajecs, n_sizes + 1))

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(settings.loc, hypothesis, "sizes"),
        exist_ok=True
    )
    circfr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"circfr_sizes.pkl"
    )
    sqfr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"sqfr_sizes.pkl"
    )
    circhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"circhex_sizes.pkl"
    )
    sqhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"sqhex_sizes.pkl"
    )
    circpathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"circpathhex_sizes.pkl"
    )
    sqpathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"sqpathhex_sizes.pkl"
    )

    # if firing rate array pickle files don't exist, run simulations
    if not os.path.exists(circfr_fname) and not os.path.exists(sqfr_fname):
        # iterate over saved trajectories
        # if not using fixed offsets, generate new set for each trajectory
        # if not os.path.exists(saveloc):
        # simulate grid cell activity for each trajectory
        for i in range(num_trajecs):
            if hypothesis == "repsupp" or hypothesis == "conjunctive":
                ox, oy = gen_offsets()
            elif hypothesis == "clustering":
                ox, oy = gen_offsets(kappacl=settings.kappa_sr)
            oxr, oyr = convert_to_rhombus(ox, oy)
            if i >= num_trajecs:
                break
            if i == 1:
                start_time = time.monotonic()
            saveloc = os.path.join(
                settings.loc,
                hypothesis,
                "sizes",
                f"{traj_type}",
                f"{i}"
            )
            for j, size in enumerate(sizes):
                if j == 1:
                    start_time2 = time.monotonic()
                circle_trajec = traj(
                    dt=settings.dt,
                    tmax=settings.tmax,
                    sp=settings.speed,
                    init_dir=settings.init_dir,
                    dphi=settings.dphi,
                    bound=size,
                    sq_bound=False
                )
                square_trajec = traj(
                    dt=settings.dt,
                    tmax=settings.tmax,
                    sp=settings.speed,
                    init_dir=settings.init_dir,
                    dphi=settings.dphi,
                    bound=size,
                    sq_bound=True
                )
                trajecs = [circle_trajec, square_trajec]
                for k, trajec in enumerate(trajecs):
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
                            propconj=settings.propconj_r,
                            kappa=settings.kappac_r,
                            jitter=settings.jitterc_r
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

                    if k == 0:
                        circfr_arr[i, j, :] = mean_fr
                        circhexes[i, j] = get_hexsym(
                            summed_fr,
                            trajec
                        )
                        circpathhexes[i, j] = get_pathsym(
                            trajec
                        )
                    elif k == 1:
                        sqfr_arr[i, j, :] = mean_fr
                        sqhexes[i, j] = get_hexsym(
                            summed_fr,
                            trajec
                        )
                        sqpathhexes[i, j] = get_pathsym(
                            trajec
                        )
                    # trajtype = "circle" if k == 0 else "square"
                    # fname = os.path.join(
                    #     saveloc,
                    #     f"{trajtype}{size}.pkl"
                    # )
                    # os.makedirs(
                    #     os.path.dirname(fname),
                    #     exist_ok=True
                    # )
                    # with open(fname, "wb") as f:
                    #     pickle.dump(mean_fr, f)

                if j == 1:
                    end_time2 = time.monotonic()
                    print(
                        "One size simulation finished in: ", 
                        timedelta(seconds=end_time2 - start_time2)
                    )
                    print(
                        f"{n_sizes} simulations estimated in: ", 
                        timedelta(seconds=(end_time2 - start_time2) * n_sizes)
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

        os.makedirs(os.path.dirname(circfr_fname), exist_ok=True)
        os.makedirs(os.path.dirname(sqfr_fname), exist_ok=True)
        os.makedirs(os.path.dirname(circhex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(sqhex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(circpathhex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(sqpathhex_fname), exist_ok=True)
        with open(circfr_fname, "wb") as f:
            pickle.dump(circfr_arr, f)
        with open(sqfr_fname, "wb") as f:
            pickle.dump(sqfr_arr, f)
        with open(circhex_fname, "wb") as f:
            pickle.dump(circhexes, f)
        with open(sqhex_fname, "wb") as f:
            pickle.dump(sqhexes, f)
        with open(circpathhex_fname, "wb") as f:
            pickle.dump(circpathhexes, f)
        with open(sqpathhex_fname, "wb") as f:
            pickle.dump(sqpathhexes, f)
    else:
        print("sizes simulations already exist")

    print("Generating plots for simulated sizes data...")

    # load arrays for plotting
    circfr_arr = np.load(circfr_fname, allow_pickle=True)
    sqfr_arr = np.load(sqfr_fname, allow_pickle=True)
    circhexes = np.load(circhex_fname, allow_pickle=True)
    sqhexes = np.load(sqhex_fname, allow_pickle=True)
    circpathhexes = np.load(circpathhex_fname, allow_pickle=True)
    sqpathhexes = np.load(sqpathhex_fname, allow_pickle=True)

    # calculate path-normalised hexasymmetry:
    # trajectories with path hexasymmetries an order of magnitude smaller than 
    # the mean are discarded as outliers
    circhexes_normed = np.delete(
        circhexes / circpathhexes,
        np.where(circpathhexes < 0.001)[0],
        axis=0
    )
    sqhexes_normed = np.delete(
        sqhexes / sqpathhexes,
        np.where(sqpathhexes < 0.001)[0],
        axis=0
    )

    # plot firing rates
    for idx, size in enumerate(sizes): 
        frplotloc = os.path.join(
            settings.loc,
            "plots",
            hypothesis,
            "sizes",
            f"{traj_type}",
            f"fr_{int(size)}.png"
        )
        direc_binned = np.linspace(-np.pi, np.pi, phbins + 1)
        angles = 180. / np.pi * \
            (direc_binned[:-1] + direc_binned[1:]) / 2.
        # fr_mean = normalise_fr(summed_fr, direc_binned, traj)
        plt.figure(figsize=(12, 4))
        plt.rcParams.update({'font.size': settings.fs})
        plt.plot(
            angles,
            np.mean(circfr_arr[:, idx, :], axis=0),
            label=f"circle boundary"
        )
        plt.plot(
            angles,
            np.mean(sqfr_arr[:, idx, :], axis=0),
            label=f"square boundary"
        )
        plt.ylabel("firing rate (Spikes/s)")
        plt.xlabel("running direction (degrees)")
        # plt.legend(prop={'size': 14})
        plt.tight_layout()
        os.makedirs(os.path.dirname(frplotloc), exist_ok=True)
        plt.savefig(frplotloc)
        plt.close()

    # plot hexsyms
    hexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"hex.png"
    )
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': settings.fs})
    plt.plot(
        sizes[:-1],
        np.mean(circhexes[:, :-1], axis=0),
        marker="^",
        lw=2,
        markersize=8,
        label="circular boundary"
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(circhexes[:, :-1], axis=0) -
        np.std(circhexes[:, :-1], axis=0) / np.sqrt(circhexes.shape[0]),
        np.mean(circhexes[:, :-1], axis=0) +
        np.std(circhexes[:, :-1], axis=0) / np.sqrt(circhexes.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        sizes[:-1],
        np.mean(sqhexes[:, :-1], axis=0),
        marker="o",
        lw=2,
        markersize=8,
        label="square boundary"
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(sqhexes[:, :-1], axis=0) -
        np.std(sqhexes[:, :-1], axis=0) / np.sqrt(sqhexes.shape[0]),
        np.mean(sqhexes[:, :-1], axis=0) +
        np.std(sqhexes[:, :-1], axis=0) / np.sqrt(sqhexes.shape[0]),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.title(hypothesis)
    plt.xlabel("Boundary size (cm)")
    plt.ylabel("Hexasymmetry (a.u.)")
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    plt.savefig(hexplotloc)
    plt.close()

    # plot normalised hexsyms
    normedhexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"normedhex.png"
    )
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': settings.fs})
    plt.plot(
        sizes[:-1],
        np.mean(circhexes_normed[:, :-1], axis=0),
        marker="^",
        lw=2,
        markersize=8,
        label="circular boundary"
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(circhexes_normed[:, :-1], axis=0) -
        np.std(circhexes_normed[:, :-1], axis=0) / np.sqrt(circhexes_normed.shape[0]),
        np.mean(circhexes_normed[:, :-1], axis=0) +
        np.std(circhexes_normed[:, :-1], axis=0) / np.sqrt(circhexes_normed.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        sizes[:-1],
        np.mean(sqhexes_normed[:, :-1], axis=0),
        marker="o",
        lw=2,
        markersize=8,
        label="square boundary"
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(sqhexes_normed[:, :-1], axis=0) -
        np.std(sqhexes_normed[:, :-1], axis=0) / np.sqrt(sqhexes_normed.shape[0]),
        np.mean(sqhexes_normed[:, :-1], axis=0) +
        np.std(sqhexes_normed[:, :-1], axis=0) / np.sqrt(sqhexes_normed.shape[0]),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.title(hypothesis)
    plt.xlabel("Boundary size (cm)")
    plt.ylabel("Hexasymmetry (norm) (a.u.)")
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    plt.savefig(normedhexplotloc)
    plt.close()

    # get offsets plot
    offloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "sizes",
        f"{traj_type}",
        "offsets.png"
    )
    oxr, oyr = get_offsets(hypothesis, tiled=True, N=settings.N)
    plt.figure(figsize=(10, 6))
    plt.scatter(oxr, oyr, s=40, c="k")
    plt.scatter(oxr[offs_idx], oyr[offs_idx], s=1000, c="red")
    plt.ioff()
    plt.axis('off')
    os.makedirs(os.path.dirname(offloc), exist_ok=True)
    plt.savefig(offloc)
    plt.close()

    # get path symmetry plot
    pathhexplotloc = os.path.join(
        settings.loc,
        "plots",
        hypothesis,
        "sizes",
        f"{traj_type}",
        f"pathhex.png"
    )
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': 18})
    plt.plot(
        sizes[:-1],
        np.mean(circpathhexes[:, :-1], axis=0),
        marker="^",
        lw=2,
        markersize=8,
        label="circular boundary"
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(circpathhexes[:, :-1], axis=0) -
        np.std(circpathhexes[:, :-1], axis=0) / np.sqrt(circpathhexes.shape[0]),
        np.mean(circpathhexes[:, :-1], axis=0) +
        np.std(circpathhexes[:, :-1], axis=0) / np.sqrt(circpathhexes.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        sizes[:-1],
        np.mean(sqpathhexes[:, :-1], axis=0),
        marker="o",
        lw=2,
        markersize=8,
        label="square boundary"
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(sqpathhexes[:, :-1], axis=0) -
        np.std(sqpathhexes[:, :-1], axis=0) / np.sqrt(sqpathhexes.shape[0]),
        np.mean(sqpathhexes[:, :-1], axis=0) +
        np.std(sqpathhexes[:, :-1], axis=0) / np.sqrt(sqpathhexes.shape[0]),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.xlabel("boundary size (cm)")
    plt.ylabel("path hexasymmetry (a.u.)")
    plt.legend(prop={'size': 14})
    plt.tight_layout()

    os.makedirs(os.path.dirname(pathhexplotloc), exist_ok=True)
    plt.savefig(pathhexplotloc)
    plt.close()


if __name__ == "__main__":
    start_time = time.monotonic()
    for traj_type in ["rw"]:
        for hypothesis in ["conjunctive", "clustering", "repsupp"]:
            sim_bnds(
                traj_type=traj_type,
                num_trajecs=settings.ntrials_finite,
                hypothesis=hypothesis
            )
    end_time = time.monotonic()
    print(
        "Sizes simulations finished in: ", 
        timedelta(seconds=end_time - start_time)
    )

