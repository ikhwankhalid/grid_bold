# %%
from functions.gridfcts import gridpop, gen_offsets
from utils.data_handler import load_data
from utils.utils import (
    visualise_traj,
    convert_to_rhombus,
    get_hexsym,
    path_hexsym
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import pickle
import os
import utils.settings as settings


def conj_bnds(
    N: int = settings.N,
    grsc: float = settings.grsc,
    phbins: int = settings.phbins,
    kappa: float = settings.kappa,
    jitter: float = settings.jitter,
    propconj: int = settings.propconj_r,
    bounds: list = [],
    idx: int = 50,
    num_trajecs: int = 20,
):
    # sample and save offsets
    if not os.path.exists(settings.offs_fname):
        print("Sampling offsets...")
        ox, oy = gen_offsets(N=N, kappacl=0.)
        ox, oy = convert_to_rhombus(ox, oy)
        os.makedirs(os.path.dirname(settings.offs_fname), exist_ok=True)
        with open(settings.offs_fname, "wb") as f:
            pickle.dump([ox, oy], f)
    else:
        with open(settings.offs_fname, "rb") as f:
            print("Loading offsets from existing file...")
            ox, oy = pickle.load(f)

    # initialise hexasymmetry array
    square_hexsym = np.zeros((len(os.listdir(settings.traj_folders)), len(bounds)))
    circle_hexsym = np.zeros((len(os.listdir(settings.traj_folders)), len(bounds)))
    square_pathsym = np.zeros((len(os.listdir(settings.traj_folders)), len(bounds)))
    circle_pathsym = np.zeros((len(os.listdir(settings.traj_folders)), len(bounds)))
    hexplotloc = os.path.join(
        settings.loc,
        "plots",
        "conjunctive",
        "boundaries",
        f"hex.png"
    )
    square_sort_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "boundaries",
        f"square_hexsym.pkl"
    )
    circle_sort_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "boundaries",
        f"circle_hexsym.pkl"
    )
    pathhexplotloc = os.path.join(
        settings.loc,
        "plots",
        "conjunctive",
        "boundaries",
        f"pathhex.png"
    )
    square_path_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "boundaries",
        f"square_pathsym.pkl"
    )
    circle_path_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "boundaries",
        f"circle_pathsym.pkl"
    )

    # iterate over trajectory folders
    for i, set_folder in enumerate(os.listdir(settings.traj_folders)):
        trajloc = os.path.join(settings.traj_folders, set_folder)
        saveloc = os.path.join(
            settings.loc,
            "conjunctive",
            "boundaries",
            set_folder
        )

        # simulate grid cell activity for each trajectory
        with os.scandir(trajloc) as it:
            for entry in it:
                if entry.name.endswith(".pkl") and entry.is_file():
                    print("loading trajectory...")
                    trajname = entry.name.split(".")[0]
                    traj = np.load(entry.path, allow_pickle=True)
                    fname = os.path.join(
                        saveloc,
                        f"{trajname}_unctive_bnds.pkl"
                    )
                    if not os.path.exists(fname):
                        direc_binned, fr_mean, fr, summed_fr = gridpop(
                            N,
                            grsc,
                            phbins,
                            traj,
                            ox,
                            oy,
                            conj=True,
                            kappa=kappa,
                            jitter=jitter,
                            propconj=propconj
                        )
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        with open(fname, "wb") as f:
                            pickle.dump((direc_binned, summed_fr, fr_mean), f)
                    else:
                        print(
                            "Firing rate data exists, skipping simulation."
                        )

                    # get and save hexasymmetry
                    hexsym_fname = os.path.join(
                        saveloc,
                        "hexsym",
                        f"{trajname}_hexsym.pkl"
                    )
                    pathsym_fname = os.path.join(
                        saveloc,
                        "pathsym",
                        f"{trajname}_pathsym.pkl"
                    )
                    if not os.path.exists(hexsym_fname):
                        os.makedirs(os.path.dirname(hexsym_fname), exist_ok=True)
                        os.makedirs(os.path.dirname(pathsym_fname), exist_ok=True)
                        direc_binned, summed_fr, fr_mean = np.load(
                            fname,
                            allow_pickle=True
                        )
                        # fr_mean = normalise_fr(summed_fr, direc_binned, traj)
                        hexsym = get_hexsym(summed_fr, traj, direc_binned)
                        pathsym = path_hexsym(traj, direc_binned)
                        with open(hexsym_fname, "wb") as f:
                            pickle.dump(hexsym, f)
                        with open(pathsym_fname, "wb") as f:
                            pickle.dump(pathsym, f)

        print("Generating plots for simulated conj data...")

        # get trajectories plot
        with os.scandir(trajloc) as it:
            for entry in it:
                if entry.name.endswith(".pkl") and entry.is_file():
                    trajname = entry.name.split(".")[0]
                    traj = np.load(entry.path, allow_pickle=True)

                    visualise_traj(
                        traj,
                        fname=os.path.join(f"sims/{set_folder}",f"{trajname}_grid"),
                        offs=[ox[idx], oy[idx]]
                    )

        # get hexasymmetries
        if (
            not os.path.exists(square_sort_fname) or
                not os.path.exists(circle_sort_fname)):
            hex_square = []
            square_bnd = []
            hex_circle = []
            circle_bnd = []
            with os.scandir(
                os.path.join(saveloc, "hexsym")
            ) as it:
                for entry in it:
                    if entry.is_file():
                        hex = np.load(entry.path, allow_pickle=True)
                        bndval = int(entry.name.split("_")[1])
                        if entry.name.split("_")[0] == "circle":
                            circle_bnd.append(bndval)
                            hex_circle.append(hex)
                        elif entry.name.split("_")[0] == "square":
                            square_bnd.append(bndval)
                            hex_square.append(hex)

            # sort hexasymmetries
            hex_square = np.array(hex_square)
            square_bnd = np.array(square_bnd)
            hex_circle = np.array(hex_circle)
            circle_bnd = np.array(circle_bnd)
            hex_square = hex_square[square_bnd.argsort()]
            hex_circle = hex_circle[circle_bnd.argsort()]

            # put sorted hexasymmetries into array
            square_hexsym[i, :] = hex_square
            circle_hexsym[i, :] = hex_circle

        # get path hexasymmetries
        if (
            not os.path.exists(square_path_fname) or
                not os.path.exists(circle_path_fname)):
            path_hex_square = []
            square_bnd = []
            path_hex_circle = []
            circle_bnd = []
            with os.scandir(
                os.path.join(saveloc, "pathsym")
            ) as it:
                for entry in it:
                    if entry.is_file():
                        hex = np.load(entry.path, allow_pickle=True)
                        bndval = int(entry.name.split("_")[1])
                        if entry.name.split("_")[0] == "circle":
                            circle_bnd.append(bndval)
                            path_hex_circle.append(hex)
                        elif entry.name.split("_")[0] == "square":
                            square_bnd.append(bndval)
                            path_hex_square.append(hex)

            # sort path hexasymmetries
            path_hex_square = np.array(path_hex_square)
            square_bnd = np.array(square_bnd)
            path_hex_circle = np.array(path_hex_circle)
            circle_bnd = np.array(circle_bnd)
            path_hex_square = path_hex_square[square_bnd.argsort()]
            path_hex_circle = path_hex_circle[circle_bnd.argsort()]

            # put sorted path hexasymmetries into array
            square_pathsym[i, :] = path_hex_square
            circle_pathsym[i, :] = path_hex_circle

        # load and plot firing rates
        with os.scandir(os.path.join(saveloc)) as it:
            for entry in it:
                if entry.name.endswith("bnds.pkl") and entry.is_file():
                    prefix = entry.name.split("_")
                    frplotloc = os.path.join(
                        settings.loc,
                        "plots",
                        "conjunctive",
                        "boundaries",
                        f"{set_folder}",
                        f"{prefix[0]}{prefix[1]}_rates.png"
                    )
                    if not os.path.exists(frplotloc):
                        direc_binned, summed_fr, fr_mean = np.load(
                            entry.path,
                            allow_pickle=True
                        )
                        angles = 180. / np.pi * \
                            (direc_binned[:-1] + direc_binned[1:]) / 2.
                        # fr_mean = normalise_fr(summed_fr, direc_binned, traj)
                        plt.figure(figsize=(12, 4))
                        plt.rcParams.update({'font.size': 18})
                        plt.plot(
                            angles,
                            fr_mean,
                            label=f"{prefix[0]} boundary {prefix[1]}"
                        )
                        plt.ylabel("firing rate (Spikes/s)")
                        plt.xlabel("running direction (degrees)")
                        # plt.legend()
                        plt.tight_layout()
                        os.makedirs(os.path.dirname(frplotloc), exist_ok=True)
                        plt.savefig(frplotloc)
                        plt.close()
                    else:
                        print(
                            f"firing rate plot exists for {prefix[0]}",
                            f" boundary {prefix[1]}, skipping."
                        )

    # save final hexasymmetry array
    if (
        not os.path.exists(square_sort_fname) or
            not os.path.exists(circle_sort_fname)):
        with open(square_sort_fname, "wb") as f:
            pickle.dump(square_hexsym, f)
        with open(circle_sort_fname, "wb") as f:
            pickle.dump(circle_hexsym, f)
    else:
        print("hexasymmetry data exists, plotting...")
        square_hexsym = np.load(square_sort_fname, allow_pickle=True)
        circle_hexsym = np.load(circle_sort_fname, allow_pickle=True)

    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': 18})
    plt.plot(
        bounds[:-1],
        np.mean(square_hexsym[:, :-1], axis=0),
        marker="o",
        lw=2,
        markersize=8,
        label="square boundary"
    )
    plt.fill_between(
        bounds[:-1],
        np.mean(square_hexsym[:, :-1], axis=0) -
        np.std(square_hexsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        np.mean(square_hexsym[:, :-1], axis=0) +
        np.std(square_hexsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        alpha=0.2,
    )
    plt.plot(
        bounds[:-1],
        np.mean(circle_hexsym[:, :-1], axis=0),
        marker="^",
        lw=2,
        markersize=8,
        label="circular boundary"
    )
    plt.fill_between(
        bounds[:-1],
        np.mean(circle_hexsym[:, :-1], axis=0) -
        np.std(circle_hexsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        np.mean(circle_hexsym[:, :-1], axis=0) +
        np.std(circle_hexsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        alpha=0.2,
    )
    plt.margins(0.01, 0.15)
    plt.xlabel("boundary size (cm)")
    plt.ylabel("hexasymmetry (a.u.)")
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    os.makedirs(os.path.dirname(hexplotloc), exist_ok=True)
    plt.savefig(hexplotloc)
    plt.close()

    # get offsets plot
    offloc = os.path.join(
        settings.loc,
        "plots",
        "conjunctive",
        "boundaries",
        "offsets.png"
    )
    ox, oy = np.load(settings.offs_fname, allow_pickle=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(ox, oy, s=40, c="k")
    plt.scatter(ox[idx], oy[idx], s=1000, c="red")
    plt.ioff()
    plt.axis('off')
    os.makedirs(os.path.dirname(offloc), exist_ok=True)
    plt.savefig(offloc)
    plt.close()

    # get path symmetry plot
    if (
        not os.path.exists(square_path_fname) or
            not os.path.exists(circle_path_fname)):
        with open(square_path_fname, "wb") as f:
            pickle.dump(square_pathsym, f)
        with open(circle_path_fname, "wb") as f:
            pickle.dump(circle_pathsym, f)
    else:
        print("Path hexasymmetry data exists, plotting...")
        square_pathsym = np.load(square_path_fname, allow_pickle=True)
        circle_pathsym = np.load(circle_path_fname, allow_pickle=True)

    plt.figure(figsize=(12, 4))
    plt.rcParams.update({'font.size': 18})
    plt.plot(
        bounds[:-1],
        np.mean(square_pathsym[:, :-1], axis=0),
        marker="o",
        lw=2,
        markersize=8,
        label="square boundary"
    )
    plt.fill_between(
        bounds[:-1],
        np.mean(square_pathsym[:, :-1], axis=0) -
        np.std(square_pathsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        np.mean(square_pathsym[:, :-1], axis=0) +
        np.std(square_pathsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        alpha=0.2,
    )
    plt.plot(
        bounds[:-1],
        np.mean(circle_pathsym[:, :-1], axis=0),
        marker="^",
        lw=2,
        markersize=8,
        label="circular boundary"
    )
    plt.fill_between(
        bounds[:-1],
        np.mean(circle_pathsym[:, :-1], axis=0) -
        np.std(circle_pathsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
        np.mean(circle_pathsym[:, :-1], axis=0) +
        np.std(circle_pathsym[:, :-1], axis=0) / np.sqrt(num_trajecs),
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
    conj_bnds()
