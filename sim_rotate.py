"""
Performs rotation simulations for all hypotheses.
"""
from functions.gridfcts import (
    gridpop_clustering,
    gridpop_repsupp,
    gridpop_conj,
    rotate_traj,
    gen_offsets,
    traj
)
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import utils.settings as settings
import time
from datetime import timedelta
from numba import get_num_threads, set_num_threads


def plt_rot_fr(fr, idxs, deg_list, fname, xlim=None, ylim=None, size="60"):
    """
    plots the firing rate as a function of movement direction for a set of
    rotated trajectories
    """
    direc_binned = np.linspace(-np.pi, np.pi, utils.settings.phbins + 1)
    angles = 180. / np.pi * \
                (direc_binned[:-1] + direc_binned[1:]) / 2.
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': utils.settings.fs})
    for idx in idxs:
        plt.plot(
            angles,
            fr[:, idx, :].mean(axis=0),
            label=f"{deg_list[idx]:.0f}$^o$"
        )
        plt.fill_between(
            angles,
            fr[:, idx, :].mean(axis=0) -
            np.std(fr[:, idx, :], axis=0) / np.sqrt(fr.shape[0]),
            fr[:, idx, :].mean(axis=0) +
            np.std(fr[:, idx, :], axis=0) / np.sqrt(fr.shape[0]),
            alpha=0.2,
        )
    if not xlim:
        plt.xticks(np.arange(-180, 180 + 30, 30))
        title = f"circle, size={size}" if "circ" in fname else f"square, size={size}"
        plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.xlabel("Movement direction (degrees)")
    plt.ylabel("Total firing rate (spks/s)")
    if xlim:
        plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def compare_rotate_samegrid(
    traj_type: str,
    n_trajecs: int = utils.settings.ntrials_finite,
    n_ang: int = utils.settings.n_ang_rotate,
    ang_range: list = utils.settings.ang_range_rotate,
    grsc: float = utils.settings.grsc,
    N: int = utils.settings.N,
    phbins: int = utils.settings.phbins,
    hypothesis: str = None,
    size: float = 60.,
    overwrite: bool = False
):
    """_summary_

    Args:
        n_trajecs (int, optional): _description_. Defaults to settings.ntrials_finite.
        n_ang (int, optional): _description_. Defaults to settings.n_ang_rotate.
        ang_range (list, optional): _description_. Defaults to [0, np.pi].
        grsc (float, optional): _description_. Defaults to settings.grsc.
        N (int, optional): _description_. Defaults to settings.N.
        phbins (int, optional): _description_. Defaults to settings.phbins.
        hypothesis (str): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # check for valid hypothesis argument
    assert hypothesis in ["repsupp", "clustering", "conjunctive"], \
        "hypothesis must be 'repsupp', 'clustering', or 'conjunctive'"
    print("performing rotation simulations for ", hypothesis)
    
    circfr_arr = np.zeros((n_trajecs, n_ang, phbins))
    sqfr_arr = np.zeros((n_trajecs, n_ang, phbins))
    circhexes = np.zeros((n_trajecs, n_ang))
    sqhexes = np.zeros((n_trajecs, n_ang))
    circpathhexes = np.zeros((n_trajecs, n_ang))
    sqpathhexes = np.zeros((n_trajecs, n_ang))

    # create list of angles from desired range
    ang_list = np.linspace(ang_range[0], ang_range[1], num=n_ang)

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(utils.settings.loc, hypothesis, "rotate"),
        exist_ok=True
    )
    circfr_fname = os.path.join(
        utils.settings.loc,
        hypothesis,
        "rotate",
        f"{traj_type}",
        f"circfr_rotate_{size}.pkl"
    )
    sqfr_fname = os.path.join(
        utils.settings.loc,
        hypothesis,
        "rotate",
        f"{traj_type}",
        f"sqfr_rotate_{size}.pkl"
    )
    circhex_fname = os.path.join(
        utils.settings.loc,
        hypothesis,
        "rotate",
        f"{traj_type}",
        f"circhex_rotate_{size}.pkl"
    )
    sqhex_fname = os.path.join(
        utils.settings.loc,
        hypothesis,
        "rotate",
        f"{traj_type}",
        f"sqhex_rotate_{size}.pkl"
    )
    circpathhex_fname = os.path.join(
        utils.settings.loc,
        hypothesis,
        "rotate",
        f"{traj_type}",
        f"circpathhex_rotate_{size}.pkl"
    )
    sqpathhex_fname = os.path.join(
        utils.settings.loc,
        hypothesis,
        "rotate",
        f"{traj_type}",
        f"sqpathhex_rotate_{size}.pkl"
    )

    # if directory exists, assume data already exists and load
    if not os.path.exists(circfr_fname) or overwrite:
    # if not os.listdir(os.path.join(settings.loc, hypothesis, "rotate")):
        # if data does not exist, load trajectories and begin population 
        # simulations
        for i in range(n_trajecs):
            if i == 1:
                start_time = time.monotonic()
            # load offsets, initialise fr and hexasymmetry arrays
            # oxr, oyr = get_offsets(hypothesis, tiled=True)
            if hypothesis == "repsupp" or hypothesis == "conjunctive":
                ox, oy = gen_offsets(N=N, kappacl=0.)
            elif hypothesis == "clustering":
                ox, oy = gen_offsets(N=utils.settings.N, kappacl=utils.settings.kappa_sr)
            oxr, oyr = convert_to_rhombus(ox, oy)
            saveloc = os.path.join(
                utils.settings.loc,
                hypothesis,
                "rotate",
                f"{traj_type}",
                f"{i}",
            )
            circle_trajec = traj(
                dt=utils.settings.dt,
                tmax=utils.settings.tmax,
                sp=utils.settings.speed,
                init_dir=utils.settings.init_dir,
                dphi=utils.settings.dphi,
                bound=size,
                sq_bound=False
            )
            square_trajec = traj(
                dt=utils.settings.dt,
                tmax=utils.settings.tmax,
                sp=utils.settings.speed,
                init_dir=utils.settings.init_dir,
                dphi=utils.settings.dphi,
                bound=size,
                sq_bound=True
            )
            trajecs = [circle_trajec, square_trajec]
            for k, trajec in enumerate(trajecs):
                for j, ang in enumerate(ang_list):
                    if j == 1:
                        start_time2 = time.monotonic()
                    if hypothesis == "clustering":
                        _, mean_fr, _, summed_fr = gridpop_clustering(
                            N,
                            grsc,
                            phbins,
                            traj=rotate_traj(trajec=trajec, theta=ang),
                            oxr=oxr,
                            oyr=oyr
                        )
                    elif hypothesis == "conjunctive":
                        _, mean_fr, _, summed_fr = gridpop_conj(
                            N,
                            grsc,
                            phbins,
                            traj=rotate_traj(trajec=trajec, theta=ang),
                            oxr=oxr,
                            oyr=oyr,
                            propconj=utils.settings.propconj_r,
                            kappa=utils.settings.kappac_r,
                            jitter=utils.settings.jitterc_r
                        )
                    else:
                        _, mean_fr, _, summed_fr = gridpop_repsupp(
                            N,
                            grsc,
                            phbins,
                            traj=rotate_traj(trajec=trajec, theta=ang),
                            oxr=oxr,
                            oyr=oyr
                        )
                    if k == 0:
                        circfr_arr[i, j, :] = mean_fr
                        circhexes[i, j] = get_hexsym(
                            summed_fr,
                            rotate_traj(trajec=trajec, theta=ang)
                        )
                        circpathhexes[i, j] = get_pathsym(
                            rotate_traj(trajec=trajec, theta=ang)
                        )
                    elif k == 1:
                        sqfr_arr[i, j, :] = mean_fr
                        sqhexes[i, j] = get_hexsym(
                            summed_fr,
                            rotate_traj(trajec=trajec, theta=ang)
                        )
                        sqpathhexes[i, j] = get_pathsym(
                            rotate_traj(trajec=trajec, theta=ang)
                        )
                    # trajtype = "circle" if k == 0 else "square"
                    # fname = os.path.join(
                    # saveloc,
                    # f"{trajtype}{size}_{np.rad2deg(ang):.0f}.pkl"
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
                            "One rotation simulation finished in: ", 
                            timedelta(seconds=end_time2 - start_time2)
                        )
                        print(
                            f"{n_ang} simulations estimated in: ", 
                            timedelta(
                                seconds=(end_time2 - start_time2) * n_ang
                            )
                        )
            # else:
            #     print(f"{set_folder} simulation already exists")
            if i == 1:
                end_time = time.monotonic()
                print(
                    "One simulation finished in: ", 
                    timedelta(seconds=end_time - start_time)
                )
                print(
                    f"{n_trajecs} simulations estimated in: ", 
                    timedelta(seconds=(end_time - start_time) * n_trajecs)
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
        print("rotate simulations already exist")

    # plotting
    deg_list = np.rad2deg(ang_list)
    rot_folders = os.path.join(utils.settings.loc, hypothesis, "rotate")
    idxs = [0, 1, 2, 3]
    circfr = pickle.load(open(circfr_fname, "rb"))
    sqfr = pickle.load(open(sqfr_fname, "rb"))
    circhexes = pickle.load(open(circhex_fname, "rb"))
    sqhexes = pickle.load(open(sqhex_fname, "rb"))
    circpathhexes = np.load(circpathhex_fname, allow_pickle=True)
    sqpathhexes = np.load(sqpathhex_fname, allow_pickle=True)
    direc_binned = np.linspace(-np.pi, np.pi, utils.settings.phbins + 1)
    angles = 180. / np.pi * \
                (direc_binned[:-1] + direc_binned[1:]) / 2.

    # calculate path-normalised hexasymmetry
    circhexes_normed = circhexes / circpathhexes
    sqhexes_normed = sqhexes / sqpathhexes

    plt_rot_fr(
        circfr,
        idxs,
        deg_list,
        os.path.join(
            utils.settings.loc,
            "plots",
            hypothesis,
            "rotate",
            f"{traj_type}",
            f"circfr_{size}_rotate_plot.png"
        ),
        size=size
    )
    plt_rot_fr(
        sqfr,
        idxs,
        deg_list,
        os.path.join(
            utils.settings.loc,
            "plots",
            hypothesis,
            "rotate",
            f"{traj_type}",
            f"sqfr_{size}_rotate_plot.png"
        ),
        size=size
    )

    # plot hexasymmetry
    plt.figure(figsize = (12, 4))
    plt.rcParams.update({'font.size': utils.settings.fs})
    plt.plot(
        deg_list,
        circhexes.mean(axis=0),
        label=f"circle_{size}"
    )
    plt.fill_between(
        deg_list,
        circhexes.mean(axis=0) -
        np.std(circhexes, axis=0) / np.sqrt(circhexes.shape[0]),
        circhexes.mean(axis=0) +
        np.std(circhexes, axis=0) / np.sqrt(circhexes.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        deg_list,
        sqhexes.mean(axis=0),
        label=f"square_{size}"
    )
    plt.fill_between(
        deg_list,
        sqhexes.mean(axis=0) -
        np.std(sqhexes, axis=0) / np.sqrt(sqhexes.shape[0]),
        sqhexes.mean(axis=0) +
        np.std(sqhexes, axis=0) / np.sqrt(sqhexes.shape[0]),
        alpha=0.2,
    )
    plt.legend(prop={'size': 14})
    plt.title(hypothesis)
    plt.tight_layout()
    plt.xlabel("Rotation angle (degrees)")
    plt.xticks(np.linspace(0, 180, 19))
    plt.ylabel("Hexasymmetry (a.u.)")
    plt.savefig(
        os.path.join(
            utils.settings.loc,
            "plots",
            hypothesis,
            "rotate",
            f"{traj_type}",
            f"hex_{size}_rotate.png"
        )
    )

    # plot path-normalised hexasymmetry
    plt.figure(figsize = (12, 4))
    plt.rcParams.update({'font.size': utils.settings.fs})
    plt.plot(
        deg_list,
        circhexes_normed.mean(axis=0),
        label=f"circle_{size}"
    )
    plt.fill_between(
        deg_list,
        circhexes_normed.mean(axis=0) -
        np.std(circhexes_normed, axis=0) / np.sqrt(circhexes_normed.shape[0]),
        circhexes_normed.mean(axis=0) +
        np.std(circhexes_normed, axis=0) / np.sqrt(circhexes_normed.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        deg_list,
        sqhexes_normed.mean(axis=0),
        label=f"square_{size}"
    )
    plt.fill_between(
        deg_list,
        sqhexes_normed.mean(axis=0) -
        np.std(sqhexes_normed, axis=0) / np.sqrt(sqhexes_normed.shape[0]),
        sqhexes_normed.mean(axis=0) +
        np.std(sqhexes_normed, axis=0) / np.sqrt(sqhexes_normed.shape[0]),
        alpha=0.2,
    )
    plt.legend(prop={'size': 14})
    plt.title(hypothesis)
    plt.tight_layout()
    plt.xlabel("Rotation angle (degrees)")
    plt.xticks(np.linspace(0, 180, 19))
    plt.ylabel("Hexasymmetry (norm) (a.u.)")
    plt.savefig(
        os.path.join(
            utils.settings.loc,
            "plots",
            hypothesis,
            "rotate",
            f"{traj_type}",
            f"normedhex_{size}_rotate.png"
        )
    )

if __name__ == "__main__":
    start_time = time.monotonic()
    for traj_type in ["rw"]:
        for hypothesis in ["conjunctive", "clustering", "repsupp"]:
            for size in utils.settings.rot_sizes:
                compare_rotate_samegrid(
                    traj_type=traj_type,
                    n_trajecs=utils.settings.ntrials_finite,
                    n_ang=utils.settings.n_ang_rotate,
                    ang_range=utils.settings.ang_range_rotate,
                    N=utils.settings.N,
                    hypothesis=hypothesis,
                    size=size
                )
    end_time = time.monotonic()
    print(
        "Rotation simulations finished in: ", 
        timedelta(seconds=end_time - start_time)
    )
    