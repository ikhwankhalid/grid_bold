"""
This script compares the distribution of movement directions between
trajectories constrained to circular or square boundaries. The distributions
are plotted and saved to two differrent '.png' files located in the
/grid_bold/data/outputs/plots/trajectories/rw/ folder. The movement directions
are saved to a '.pkl' file in the same folder. This script was not used
in any of the plots of the manuscipt, but acted as a sanity check for the
distribution of movement directions in bounded environments. The first run of
the script performs 20 trials for each boundary type. Every subsequent run
loads the existing '.pkl' file and adds 20 more trials to the existing data.
"""
from utils.grid_funcs import traj
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import settings

###############################################################################
# Parameters                                                                  #
###############################################################################
tmax = settings.tmax


###############################################################################
# Functions                                                                   #
###############################################################################
def compare_rotate_samegrid(
    num_trajecs: int,
    traj_type: str,
    size: float = 60.
):
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
        direcs_dict["circle"] = np.histogram(
            np.rad2deg(circ_direc), bins=np.arange(-180, 181, 1)
        )[0]
        direcs_dict["square"] = np.histogram(
            np.rad2deg(sq_direc), bins=np.arange(-180, 181, 1)
        )[0]
        direcs_dict["bins"] = np.histogram(
            np.rad2deg(circ_direc), bins=np.arange(-180, 181, 1)
        )[1]

    else:
        direcs_dict = pickle.load(open(direcs_dict_fname, "rb"))

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
        direcs_dict["circle"] += np.histogram(
            np.rad2deg(circ_direc), bins=np.arange(-180, 181, 1)
        )[0]
        direcs_dict["square"] += np.histogram(
            np.rad2deg(sq_direc), bins=np.arange(-180, 181, 1)
        )[0]

    os.makedirs(
        saveloc,
        exist_ok=True
    )

    os.makedirs(os.path.dirname(direcs_dict_fname), exist_ok=True)
    with open(direcs_dict_fname, "wb") as f:
        pickle.dump(direcs_dict, f)

    # plotting
    bins = direcs_dict["bins"]

    # plot hexasymmetry
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': settings.fs})
    plt.bar(
        bins[:-1],
        direcs_dict["circle"] / np.sum(direcs_dict["circle"]),
        width=1.
    )
    n_circ = np.sum(direcs_dict["circle"])
    n_circ = int(n_circ / (settings.tmax / settings.dt))
    plt.title(f"circular boundary, {n_circ} trajectories")
    plt.tight_layout()
    plt.xlabel("Movement direction (degrees)")
    plt.ylabel("Probability")
    plt.savefig(
        os.path.join(
            saveloc,
            "circle_histogram.png"
        )
    )

    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': settings.fs})
    plt.bar(
        bins[:-1],
        direcs_dict["square"] / np.sum(direcs_dict["square"]),
        width=1.
    )
    n_sq = np.sum(direcs_dict["square"])
    n_sq = int(n_sq / (settings.tmax / settings.dt))
    plt.title(f"square boundary, {n_sq} trajectories")
    plt.tight_layout()
    plt.xlabel("Movement direction (degrees)")
    plt.ylabel("Probability")
    plt.savefig(
        os.path.join(
            saveloc,
            "square_histogram.png"
        )
    )


###############################################################################
# Run                                                                         #
###############################################################################
if __name__ == "__main__":
    compare_rotate_samegrid(20, "rw", 60.)
