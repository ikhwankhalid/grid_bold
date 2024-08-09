"""
This script compiles figures illustrating the effect of boundary size and
boundary rotation on the hexasymmetry for each hypothesis (supplementary
figures S5 and S6 in the manuscript) then saves the figures as .PNG files.
Sets of figures are made for different mean grid offsets (zero or random).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import settings
from utils.utils import ax_pos
from utils.grid_funcs import (
    traj,
    rotate_traj
)
from utils.utils import (
    get_plotting_grid
)
matplotlib.use('Agg')

###############################################################################
# Parameters                                                                  #
###############################################################################
lw = 3.                                   # width of line plots
yticks = [0, 600, 1200]                   # y ticks for neural H plots
ylim = [1, 6000]                          # y limits for neural H plots
path_yticks = [0.002, 0.006, 0.010]       # y ticks for path H plots
path_ylim = [0.001, 0.01]                 # y limits for path H plots
offs = [0, 0]                             # grid offset for illustrations

# boundary size parameters
sizes = np.append(                        # boundary sizes in cm
    np.linspace(
        settings.size_range[0],
        settings.size_range[1],
        settings.n_sizes
    ),
    settings.inf_size
)
size_illusts = np.array(                  # boundary sizes for illustration
    [30, 80, 130, 180]
)

# boundary rotation parameters
ang_list = np.linspace(                   # boundary rotations in rad
    settings.ang_range_rotate[0],
    settings.ang_range_rotate[1],
    num=settings.n_ang_rotate
)
deg_list = np.rad2deg(ang_list)           # boundary rotations in degrees
rot_illusts = np.array(                   # boundary rotations for illustration
    [0, 30, 60, 90]
) / 180 * np.pi


###############################################################################
# Plotting                                                                    #
###############################################################################
# meanoff_types = ["zero", "uniform"]
meanoff_types = ["zero"]
conj_orient = "zero"
for meanoff_type in meanoff_types:
    # load hexasymmetry and path hexasymmetry arrays for boundary rotation
    conjcirchex_rot_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "rotate",
        meanoff_type,
        conj_orient,
        "circhex_rotate_60.0.pkl"
    )
    conjsquarehex_rot_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "rotate",
        meanoff_type,
        conj_orient,
        "sqhex_rotate_60.0.pkl"
    )

    clustcirchex_rot_fname = os.path.join(
        settings.loc,
        "clustering",
        "rotate",
        meanoff_type,
        "zero",
        "circhex_rotate_60.0.pkl"
    )
    clustsquarehex_rot_fname = os.path.join(
        settings.loc,
        "clustering",
        "rotate",
        meanoff_type,
        "zero",
        "sqhex_rotate_60.0.pkl"
    )

    repsuppcirchex_rot_fname = os.path.join(
        settings.loc,
        "repsupp",
        "rotate",
        meanoff_type,
        "zero",
        "circhex_rotate_60.0.pkl"
    )
    repsuppsquarehex_rot_fname = os.path.join(
        settings.loc,
        "repsupp",
        "rotate",
        meanoff_type,
        "zero",
        "sqhex_rotate_60.0.pkl"
    )

    pathcirchex_rot_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "rotate",
        meanoff_type,
        "zero",
        "circpathhex_rotate_60.0.pkl"
    )
    pathsquarehex_rot_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "rotate",
        meanoff_type,
        "zero",
        "sqpathhex_rotate_60.0.pkl"
    )

    with open(conjcirchex_rot_fname, "rb") as f:
        conjcirchex_rot = pickle.load(f)
    with open(conjsquarehex_rot_fname, "rb") as f:
        conjsquarehex_rot = pickle.load(f)
    with open(clustcirchex_rot_fname, "rb") as f:
        clustcirchex_rot = pickle.load(f)
    with open(clustsquarehex_rot_fname, "rb") as f:
        clustsquarehex_rot = pickle.load(f)
    with open(repsuppcirchex_rot_fname, "rb") as f:
        repsuppcirchex_rot = pickle.load(f)
    with open(repsuppsquarehex_rot_fname, "rb") as f:
        repsuppsquarehex_rot = pickle.load(f)
    with open(pathcirchex_rot_fname, "rb") as f:
        pathcirchex_rot = pickle.load(f)
    with open(pathsquarehex_rot_fname, "rb") as f:
        pathsquarehex_rot = pickle.load(f)

    # load hexasymmetry and path hexasymmetry arrays for boundary size
    conjcirchex_size_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "sizes",
        meanoff_type,
        "zero",
        "circhex_sizes.pkl"
    )
    conjsquarehex_size_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "sizes",
        meanoff_type,
        "zero",
        "sqhex_sizes.pkl"
    )

    clustcirchex_size_fname = os.path.join(
        settings.loc,
        "clustering",
        "sizes",
        meanoff_type,
        "zero",
        "circhex_sizes.pkl"
    )
    clustsquarehex_size_fname = os.path.join(
        settings.loc,
        "clustering",
        "sizes",
        meanoff_type,
        "zero",
        "sqhex_sizes.pkl"
    )

    repsuppcirchex_size_fname = os.path.join(
        settings.loc,
        "repsupp",
        "sizes",
        meanoff_type,
        "zero",
        "circhex_sizes.pkl"
    )
    repsuppsquarehex_size_fname = os.path.join(
        settings.loc,
        "repsupp",
        "sizes",
        meanoff_type,
        "zero",
        "sqhex_sizes.pkl"
    )

    pathcirchex_size_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "sizes",
        meanoff_type,
        "zero",
        "circpathhex_sizes.pkl"
    )
    pathsquarehex_size_fname = os.path.join(
        settings.loc,
        "conjunctive",
        "sizes",
        meanoff_type,
        "zero",
        "sqpathhex_sizes.pkl"
    )

    with open(conjcirchex_size_fname, "rb") as f:
        conjcirchex_size = pickle.load(f)
    with open(conjsquarehex_size_fname, "rb") as f:
        conjsquarehex_size = pickle.load(f)
    with open(clustcirchex_size_fname, "rb") as f:
        clustcirchex_size = pickle.load(f)
    with open(clustsquarehex_size_fname, "rb") as f:
        clustsquarehex_size = pickle.load(f)
    with open(repsuppcirchex_size_fname, "rb") as f:
        repsuppcirchex_size = pickle.load(f)
    with open(repsuppsquarehex_size_fname, "rb") as f:
        repsuppsquarehex_size = pickle.load(f)

    with open(pathcirchex_size_fname, "rb") as f:
        pathcirchex_size = pickle.load(f)
    with open(pathsquarehex_size_fname, "rb") as f:
        pathsquarehex_size = pickle.load(f)

    ###########################################################################
    # Size figure                                                             #
    ###########################################################################
    # sample random walk trajectories in finite environments of different sizes
    size30_circtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot*0.6,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[0],
        sq_bound=False
    )
    size30_sqtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot*0.6,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[0],
        sq_bound=True
    )
    size80_circtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[1],
        sq_bound=False
    )
    size80_sqtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[1],
        sq_bound=True
    )
    size130_circtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot*1.5,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[2],
        sq_bound=False
    )
    size130_sqtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot*1.5,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[2],
        sq_bound=True
    )
    size180_circtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot*3,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[3],
        sq_bound=False
    )
    size180_sqtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot*3,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=size_illusts[3],
        sq_bound=True
    )

    # start plotting
    fig = plt.figure(figsize=(16, 24))
    plt.rcParams.update({'font.size': settings.fs})
    spec = fig.add_gridspec(
        ncols=4,
        nrows=6,
    )

    # illustrations of trajectories overlayed on grid fields
    ax_circsize01 = fig.add_subplot(spec[0, 0])
    get_plotting_grid(
        size30_circtrajec, offs, extent=200, title="Size 30 cm",
        titlepos=[0.5, 1.1],
        yticks=[-200, 0, 200],
        ylabel="y (cm)")
    ax_circsize02 = fig.add_subplot(spec[0, 1])
    get_plotting_grid(size80_circtrajec, offs, extent=200,
                      title="Size 80 cm", titlepos=[0.5, 1.1])
    ax_circsize03 = fig.add_subplot(spec[0, 2])
    get_plotting_grid(size130_circtrajec, offs, extent=200,
                      title="Size 130 cm", titlepos=[0.5, 1.1])
    ax_circsize04 = fig.add_subplot(spec[0, 3])
    get_plotting_grid(size180_circtrajec, offs, extent=200,
                      title="Size 180 cm", titlepos=[0.5, 1.1])

    ax_squaresize01 = fig.add_subplot(spec[1, 0])
    get_plotting_grid(
        size30_sqtrajec, offs, extent=200, yticks=[-200, 0, 200],
        xticks=[0, 200],
        ylabel="y (cm)", xlabel="x (cm)")
    ax_squaresize02 = fig.add_subplot(spec[1, 1])
    get_plotting_grid(size80_sqtrajec, offs, extent=200,
                      xticks=[-200, 0, 200], xlabel="x (cm)")
    ax_squaresize03 = fig.add_subplot(spec[1, 2])
    get_plotting_grid(size130_sqtrajec, offs, extent=200,
                      xticks=[-200, 0, 200], xlabel="x (cm)")
    ax_squaresize04 = fig.add_subplot(spec[1, 3])
    get_plotting_grid(size180_sqtrajec, offs, extent=200,
                      xticks=[-200, 0, 200], xlabel="x (cm)")

    # conjunctive
    ax_conjsize = fig.add_subplot(spec[2, :])
    ax_conjsize.spines['top'].set_visible(False)
    ax_conjsize.spines['right'].set_visible(False)
    plt.plot(
        sizes[:-1],
        np.mean(conjcirchex_size[:, :-1], axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(conjcirchex_size[:, :-1], axis=0)
        - np.std(conjcirchex_size[:, :-1], axis=0)
        / np.sqrt(conjcirchex_size.shape[0]),
        np.mean(conjcirchex_size[:, :-1], axis=0)
        + np.std(conjcirchex_size[:, :-1], axis=0)
        / np.sqrt(conjcirchex_size.shape[0]),
        alpha=0.2
    )
    plt.plot(
        sizes[:-1],
        np.mean(conjsquarehex_size[:, :-1], axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(conjsquarehex_size[:, :-1], axis=0)
        - np.std(conjsquarehex_size[:, :-1], axis=0)
        / np.sqrt(conjsquarehex_size.shape[0]),
        np.mean(conjsquarehex_size[:, :-1], axis=0)
        + np.std(conjsquarehex_size[:, :-1], axis=0)
        / np.sqrt(conjsquarehex_size.shape[0]),
        alpha=0.2
    )
    plt.margins(0.01, 0.15)
    plt.title("Conjunctive", y=0.80)
    plt.ylabel("Hexasymmetry (spk/s)")
    plt.xticks(np.linspace(30, 180, 6), [])
    plt.xlim(30, 180)
    # plt.yticks(yticks)
    plt.ylim(ylim[0], ylim[1])
    plt.yscale("log")
    plt.legend(prop={'size': 14}, loc="lower right")

    # repetition suppression
    ax_repsuppsize = fig.add_subplot(spec[3, :])
    ax_repsuppsize.spines['top'].set_visible(False)
    ax_repsuppsize.spines['right'].set_visible(False)
    plt.plot(
        sizes[:-1],
        np.mean(repsuppcirchex_size[:, :-1], axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(repsuppcirchex_size[:, :-1], axis=0)
        - np.std(repsuppcirchex_size[:, :-1], axis=0)
        / np.sqrt(repsuppcirchex_size.shape[0]),
        np.mean(repsuppcirchex_size[:, :-1], axis=0)
        + np.std(repsuppcirchex_size[:, :-1], axis=0)
        / np.sqrt(repsuppcirchex_size.shape[0]),
        alpha=0.2
    )
    plt.plot(
        sizes[:-1],
        np.mean(repsuppsquarehex_size[:, :-1], axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[: -1],
        np.mean(repsuppsquarehex_size[:, : -1], axis=0)
        - np.std(repsuppsquarehex_size[:, : -1], axis=0)
        / np.sqrt(repsuppsquarehex_size.shape[0]),
        np.mean(repsuppsquarehex_size[:, : -1], axis=0)
        + np.std(repsuppsquarehex_size[:, : -1], axis=0)
        / np.sqrt(repsuppsquarehex_size.shape[0]),
        alpha=0.2)
    plt.title("Repetition suppression", y=0.80)
    plt.xticks(np.linspace(30, 180, 6), [])
    plt.ylabel("Hexasymmetry (spk/s)")
    plt.xlim(30, 180)
    # plt.yticks(yticks)
    plt.ylim(ylim[0], ylim[1])
    plt.yscale("log")

    # structure-function mapping
    ax_clustsize = fig.add_subplot(spec[4, :])
    ax_clustsize.spines['top'].set_visible(False)
    ax_clustsize.spines['right'].set_visible(False)
    plt.plot(
        sizes[:-1],
        np.mean(clustcirchex_size[:, :-1], axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(clustcirchex_size[:, :-1], axis=0)
        - np.std(clustcirchex_size[:, :-1], axis=0)
        / np.sqrt(clustcirchex_size.shape[0]),
        np.mean(clustcirchex_size[:, :-1], axis=0)
        + np.std(clustcirchex_size[:, :-1], axis=0)
        / np.sqrt(clustcirchex_size.shape[0]),
        alpha=0.2
    )
    plt.plot(
        sizes[:-1],
        np.mean(clustsquarehex_size[:, :-1], axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(clustsquarehex_size[:, :-1], axis=0)
        - np.std(clustsquarehex_size[:, :-1], axis=0)
        / np.sqrt(clustsquarehex_size.shape[0]),
        np.mean(clustsquarehex_size[:, :-1], axis=0)
        + np.std(clustsquarehex_size[:, :-1], axis=0)
        / np.sqrt(clustsquarehex_size.shape[0]),
        alpha=0.2
    )
    plt.title("Structure-function mapping", y=0.80)
    plt.ylabel("Hexasymmetry (spk/s)")
    plt.xticks(np.linspace(30, 180, 6), [])
    plt.xlim(30, 180)
    # plt.yticks(yticks)
    plt.ylim(ylim[0], ylim[1])
    plt.yscale("log")

    # path hexasymmetry
    ax_pathsize = fig.add_subplot(spec[5, :])
    ax_pathsize.spines['top'].set_visible(False)
    ax_pathsize.spines['right'].set_visible(False)
    plt.plot(
        sizes[:-1],
        np.mean(pathcirchex_size[:, :-1], axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(pathcirchex_size[:, :-1], axis=0)
        - np.std(pathcirchex_size[:, :-1], axis=0)
        / np.sqrt(pathcirchex_size.shape[0]),
        np.mean(pathcirchex_size[:, :-1], axis=0)
        + np.std(pathcirchex_size[:, :-1], axis=0)
        / np.sqrt(pathcirchex_size.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        sizes[:-1],
        np.mean(pathsquarehex_size[:, :-1], axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        sizes[:-1],
        np.mean(pathsquarehex_size[:, :-1], axis=0)
        - np.std(pathsquarehex_size[:, :-1], axis=0)
        / np.sqrt(pathsquarehex_size.shape[0]),
        np.mean(pathsquarehex_size[:, :-1], axis=0)
        + np.std(pathsquarehex_size[:, :-1], axis=0)
        / np.sqrt(pathsquarehex_size.shape[0]),
        alpha=0.2,
    )
    plt.xticks(np.linspace(30, 180, 6))
    plt.xlim(30, 180)
    # plt.yticks(yticks)
    plt.yticks(path_yticks)
    plt.ylim(path_ylim[0], path_ylim[1])
    # plt.yscale("log")
    plt.xlabel("Boundary size (cm)")
    plt.ylabel("Path\nhexasymmetry")

    # finetuning of axes locations/sizes
    ax_pos(ax_circsize01, -0.015, 0.025, 1., 1.)
    ax_pos(ax_circsize02, -0.005, 0.025, 1., 1.)
    ax_pos(ax_circsize03, 0.005, 0.025, 1., 1.)
    ax_pos(ax_circsize04, 0.015, 0.025, 1., 1.)

    ax_pos(ax_squaresize01, -0.015, 0.025, 1., 1.)
    ax_pos(ax_squaresize02, -0.005, 0.025, 1., 1.)
    ax_pos(ax_squaresize03, 0.005, 0.025, 1., 1.)
    ax_pos(ax_squaresize04, 0.015, 0.025, 1., 1.)

    # this ensures ylabels are aligned for neatness
    fig.align_ylabels(fig.axes[1:])

    # save the figure
    savepath = os.path.join(
        settings.loc,
        "plots",
        "finite_boundaries",
        meanoff_type,
        'Figure_sizes.png'
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300)
    plt.close()

    ###########################################################################
    # Rotation figure                                                         #
    ###########################################################################
    # sample random walk trajectories in finite environments of different sizes
    rot_circtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=60.,
        sq_bound=False
    )
    rot_sqtrajec = traj(
        dt=settings.dt,
        tmax=settings.tplot,
        sp=settings.speed,
        init_dir=settings.init_dir,
        dphi=settings.dphi,
        bound=60.,
        sq_bound=True
    )

    # start plotting
    fig = plt.figure(figsize=(16, 24))
    plt.rcParams.update({'font.size': settings.fs})
    spec = fig.add_gridspec(
        ncols=4,
        nrows=6,
    )

    # illustrations of trajectories overlayed on grid fields
    ax_circrot01 = fig.add_subplot(spec[0, 0])
    get_plotting_grid(
        rot_circtrajec, offs, extent=90., yticks=[-90, 0, 90],
        title="Rotation of 0°", titlepos=[0.5, 1.1],
        ylabel="y (cm)")
    ax_circrot02 = fig.add_subplot(spec[0, 1])
    get_plotting_grid(
        rotate_traj(rot_circtrajec, rot_illusts[1]),
        offs, extent=90., title="Rotation of 30°", titlepos=[0.5, 1.1])
    ax_circrot03 = fig.add_subplot(spec[0, 2])
    get_plotting_grid(
        rotate_traj(rot_circtrajec, rot_illusts[2]),
        offs, extent=90., title="Rotation of 60°", titlepos=[0.5, 1.1])
    ax_circrot04 = fig.add_subplot(spec[0, 3])
    get_plotting_grid(
        rotate_traj(rot_circtrajec, rot_illusts[3]),
        offs, extent=90., title="Rotation of 90°", titlepos=[0.5, 1.1])

    ax_squarerot01 = fig.add_subplot(spec[1, 0])
    get_plotting_grid(
        rotate_traj(rot_sqtrajec, rot_illusts[0]),
        offs, extent=90., xticks=[-90, 0, 90],
        yticks=[-90, 0, 90],
        xlabel="x (cm)", ylabel="y (cm)")
    ax_squarerot02 = fig.add_subplot(spec[1, 1])
    get_plotting_grid(
        rotate_traj(rot_sqtrajec, rot_illusts[1]),
        offs, extent=90., xticks=[-90, 0, 90],
        xlabel="x (cm)")
    ax_squarerot03 = fig.add_subplot(spec[1, 2])
    get_plotting_grid(
        rotate_traj(rot_sqtrajec, rot_illusts[2]),
        offs, extent=90., xticks=[-90, 0, 90],
        xlabel="x (cm)")
    ax_squarerot04 = fig.add_subplot(spec[1, 3])
    get_plotting_grid(
        rotate_traj(rot_sqtrajec, rot_illusts[3]),
        offs, extent=90., xticks=[-90, 0, 90],
        xlabel="x (cm)")

    # conjunctive
    ax_conjrot = fig.add_subplot(spec[2, :])
    ax_conjrot.spines['top'].set_visible(False)
    ax_conjrot.spines['right'].set_visible(False)
    plt.plot(
        deg_list,
        np.mean(conjcirchex_rot, axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(conjcirchex_rot, axis=0) -
        np.std(conjcirchex_rot, axis=0) / np.sqrt(conjcirchex_rot.shape[0]),
        np.mean(conjcirchex_rot, axis=0) +
        np.std(conjcirchex_rot, axis=0) / np.sqrt(conjcirchex_rot.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        deg_list,
        np.mean(conjsquarehex_rot, axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(conjsquarehex_rot, axis=0)
        - np.std(conjsquarehex_rot, axis=0)
        / np.sqrt(conjsquarehex_rot.shape[0]),
        np.mean(conjsquarehex_rot, axis=0)
        + np.std(conjsquarehex_rot, axis=0)
        / np.sqrt(conjsquarehex_rot.shape[0]),
        alpha=0.2,
    )
    plt.title("Conjunctive", y=0.80)
    plt.xticks(np.linspace(0, 360, 13), [])
    plt.ylabel("Hexasymmetry (spk/s)")
    # plt.yticks(yticks)
    plt.xlim(0, 360)
    plt.yscale("log")
    plt.ylim(ylim[0], ylim[1])
    plt.legend(prop={'size': 14}, loc="lower right")

    # repetition suppression
    ax_repsupprot = fig.add_subplot(spec[3, :])
    ax_repsupprot.spines['top'].set_visible(False)
    ax_repsupprot.spines['right'].set_visible(False)
    plt.plot(
        deg_list,
        np.mean(repsuppcirchex_rot, axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(repsuppcirchex_rot, axis=0)
        - np.std(repsuppcirchex_rot, axis=0)
        / np.sqrt(repsuppcirchex_rot.shape[0]),
        np.mean(repsuppcirchex_rot, axis=0)
        + np.std(repsuppcirchex_rot, axis=0)
        / np.sqrt(repsuppcirchex_rot.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        deg_list,
        np.mean(repsuppsquarehex_rot, axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(repsuppsquarehex_rot, axis=0)
        - np.std(repsuppsquarehex_rot, axis=0)
        / np.sqrt(repsuppsquarehex_rot.shape[0]),
        np.mean(repsuppsquarehex_rot, axis=0)
        + np.std(repsuppsquarehex_rot, axis=0)
        / np.sqrt(repsuppsquarehex_rot.shape[0]),
        alpha=0.2,
    )
    plt.title("Repetition suppression", y=0.80)
    plt.xticks(np.linspace(0, 360, 13), [])
    plt.ylabel("Hexasymmetry (spk/s)")
    plt.xlim(0, 360)
    # plt.yticks(yticks)
    plt.yscale("log")
    plt.ylim(ylim[0], ylim[1])

    # structure-function mapping
    ax_clustrot = fig.add_subplot(spec[4, :])
    ax_clustrot.spines['top'].set_visible(False)
    ax_clustrot.spines['right'].set_visible(False)
    plt.plot(
        deg_list,
        np.mean(clustcirchex_rot, axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(clustcirchex_rot, axis=0) -
        np.std(clustcirchex_rot, axis=0) / np.sqrt(clustcirchex_rot.shape[0]),
        np.mean(clustcirchex_rot, axis=0) +
        np.std(clustcirchex_rot, axis=0) / np.sqrt(clustcirchex_rot.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        deg_list,
        np.mean(clustsquarehex_rot, axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(clustsquarehex_rot, axis=0)
        - np.std(clustsquarehex_rot, axis=0)
        / np.sqrt(clustsquarehex_rot.shape[0]),
        np.mean(clustsquarehex_rot, axis=0)
        + np.std(clustsquarehex_rot, axis=0)
        / np.sqrt(clustsquarehex_rot.shape[0]),
        alpha=0.2,
    )
    plt.title("Structure-function mapping", y=0.80)
    plt.xticks(np.linspace(0, 360, 13), [])
    plt.ylabel("Hexasymmetry (spk/s)")
    plt.xlim(0, 360)
    # plt.yticks(yticks)
    plt.yscale("log")
    plt.ylim(ylim[0], ylim[1])

    ax_pathrot = fig.add_subplot(spec[5, :])
    ax_pathrot.spines['top'].set_visible(False)
    ax_pathrot.spines['right'].set_visible(False)
    plt.plot(
        deg_list,
        np.mean(pathcirchex_rot, axis=0),
        label="circular boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(pathcirchex_rot, axis=0) -
        np.std(pathcirchex_rot, axis=0) / np.sqrt(pathcirchex_rot.shape[0]),
        np.mean(pathcirchex_rot, axis=0) +
        np.std(pathcirchex_rot, axis=0) / np.sqrt(pathcirchex_rot.shape[0]),
        alpha=0.2,
    )
    plt.plot(
        deg_list,
        np.mean(pathsquarehex_rot, axis=0),
        label="square boundary",
        linewidth=lw
    )
    plt.fill_between(
        deg_list,
        np.mean(pathsquarehex_rot, axis=0)
        - np.std(pathsquarehex_rot, axis=0)
        / np.sqrt(pathsquarehex_rot.shape[0]),
        np.mean(pathsquarehex_rot, axis=0)
        + np.std(pathsquarehex_rot, axis=0)
        / np.sqrt(pathsquarehex_rot.shape[0]),
        alpha=0.2,
    )
    plt.xticks(np.linspace(0, 360, 13))
    plt.xlim(0, 360)
    plt.yticks(path_yticks)
    plt.ylim(path_ylim[0], path_ylim[1])
    plt.xlabel("Rotation angle ($^\\circ$)")
    plt.ylabel("Path\nhexasymmetry")

    # finetuning of axes locations/sizes
    ax_pos(ax_circrot01, 0, 0.025, 1., 1.)
    ax_pos(ax_circrot02, 0, 0.025, 1., 1.)
    ax_pos(ax_circrot03, 0, 0.025, 1., 1.)
    ax_pos(ax_circrot04, 0, 0.025, 1., 1.)

    ax_pos(ax_squarerot01, 0, 0.025, 1., 1.)
    ax_pos(ax_squarerot02, 0, 0.025, 1., 1.)
    ax_pos(ax_squarerot03, 0, 0.025, 1., 1.)
    ax_pos(ax_squarerot04, 0, 0.025, 1., 1.)

    # this ensures ylabels are aligned for neatness
    fig.align_ylabels(fig.axes[1:-1])

    # save the figure
    savepath = os.path.join(
        settings.loc,
        "plots",
        "finite_boundaries",
        meanoff_type,
        "conj_" + conj_orient,
        'Figure_rotation.png'
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300)
    plt.close()
