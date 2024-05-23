"""
Performs simulations using the clustering hypothesis for rotated trajectories
"""
from utils.grid_funcs import traj
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import settings

tmax = settings.tmax

def compare_rotate_samegrid(
    num_trajecs: int,
    traj_type: str,
    size: float = 60.
):
    
    if traj_type == "rw":
        traj_path = settings.rw_loc
    elif traj_type == "pwl":
        traj_path = settings.pwl_loc

    # prepare firing rate and hexasymmetry filenames for saving
    saveloc = os.path.join(
        settings.loc,
        "plots",
        "trajectories",
        f"{traj_type}"
    )

    direcs_dict_fname = os.path.join(
        saveloc,
        "direcs.pkl"
    )

    if not os.path.isfile(direcs_dict_fname):
        direcs_dict = {}
        circle_dirs = np.array([])
        square_dirs = np.array([])
        # for i, set_folder in enumerate(os.listdir(traj_path)):
        #     trajloc = os.path.join(traj_path, set_folder)
        #     with os.scandir(trajloc) as it:
        #         for entry in it:
        #             # if "_60" in entry.name:
        #             if f"_{size}." in entry.name:
        #                 trajname = entry.name.split(".")[0]
        #                 t, x, y, direc = np.load(entry.path, allow_pickle=True)
        #                 if f"circle_{size}." in entry.name:
        #                     circle_dirs = np.hstack((circle_dirs, direc))
        #                 elif f"square_{size}." in entry.name:
        #                     square_dirs = np.hstack((square_dirs, direc))
        t, x, y, sq_direc = traj(
            dt=settings.dt,
            tmax=tmax,
            sp=settings.speed,
            init_dir=settings.init_dir,
            dphi=np.pi,
            bound=size,
            sq_bound=True
        )
        t, x, y, circ_direc = traj(
            dt=settings.dt,
            tmax=tmax,
            sp=settings.speed,
            init_dir=settings.init_dir,
            dphi=np.pi,
            bound=size
            # sq_bound=False
        )
        direcs_dict["circle"] = np.histogram(np.rad2deg(circ_direc), bins=np.arange(-180, 181, 1))[0]
        direcs_dict["square"] = np.histogram(np.rad2deg(sq_direc), bins=np.arange(-180, 181, 1))[0]
        direcs_dict["bins"] = np.histogram(np.rad2deg(circ_direc), bins=np.arange(-180, 181, 1))[1]

    else:
        direcs_dict = pickle.load(open(direcs_dict_fname, "rb"))
        # circle_dirs = direcs_dict["circle"]
        # square_dirs = direcs_dict["square"]
        # bins = direcs_dict["bins"]

    for i in range(num_trajecs):
        print(i)
        t, x, y, sq_direc = traj(
            dt=settings.dt,
            tmax=tmax,
            sp=settings.speed,
            init_dir=settings.init_dir,
            dphi=settings.dphi,
            bound=size,
            sq_bound=True
        )
        t, x, y, circ_direc = traj(
            dt=settings.dt,
            tmax=tmax,
            sp=settings.speed,
            init_dir=settings.init_dir,
            dphi=settings.dphi,
            bound=size
            # sq_bound=False
        )
        direcs_dict["circle"] += np.histogram(np.rad2deg(circ_direc), bins=np.arange(-180, 181, 1))[0]
        direcs_dict["square"] += np.histogram(np.rad2deg(sq_direc), bins=np.arange(-180, 181, 1))[0]

    # direcs_dict["circle"] = circle_dirs
    # direcs_dict["square"] = square_dirs

    os.makedirs(
        saveloc,
        exist_ok=True
    )

    os.makedirs(os.path.dirname(direcs_dict_fname), exist_ok=True)
    with open(direcs_dict_fname, "wb") as f:
        pickle.dump(direcs_dict, f)

    # plotting
    direc_binned = np.linspace(-np.pi, np.pi, settings.phbins + 1)
    angles = 180. / np.pi * \
                (direc_binned[:-1] + direc_binned[1:]) / 2.
    bins = direcs_dict["bins"]

    # plot hexasymmetry
    plt.figure(figsize = (12, 4))
    plt.rcParams.update({'font.size': settings.fs})
    plt.bar(bins[:-1], direcs_dict["circle"] / np.sum(direcs_dict["circle"]), width=1.) #, density = True)
    # plt.legend(prop={'size': 14})
    n_circ = np.sum(direcs_dict["circle"])
    n_circ = int(n_circ / (settings.tmax / settings.dt))
    plt.title(f"circular boundary, {n_circ} trajectories")
    plt.tight_layout()
    plt.xlabel("Movement direction (degrees)")
    plt.ylabel("Probability")
    # plt.xticks(np.arange(-180, 181, 60))
    # plt.ylabel("Hexasymmetry (a.u.)")
    plt.savefig(
        os.path.join(
            saveloc,
            f"circle_histogram.png"
        )
    )

    plt.figure(figsize = (12, 4))
    plt.rcParams.update({'font.size': settings.fs})
    plt.bar(bins[:-1], direcs_dict["square"] / np.sum(direcs_dict["square"]), width=1.) #, density = True)
    # plt.legend(prop={'size': 14})
    n_sq = np.sum(direcs_dict["square"])
    n_sq = int(n_sq / (settings.tmax / settings.dt))
    plt.title(f"square boundary, {n_sq} trajectories")
    plt.tight_layout()
    plt.xlabel("Movement direction (degrees)")
    plt.ylabel("Probability")
    # plt.xticks(np.arange(-180, 181, 60))
    # plt.ylabel("Hexasymmetry (a.u.)")
    plt.savefig(
        os.path.join(
            saveloc,
            f"square_histogram.png"
        )
    )


if __name__ == "__main__":
    compare_rotate_samegrid(20, "rw", 60.)
