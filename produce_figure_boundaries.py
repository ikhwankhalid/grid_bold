import numpy as np
import matplotlib
from utils.utils import ax_pos

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions.gridfcts import (
    traj,
    rotate_traj
)
from matplotlib.ticker import ScalarFormatter
from utils.utils import (
    get_plotting_grid
)
import utils.settings as settings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pickle


sizes = np.append(
    np.linspace(settings.size_range[0], settings.size_range[1], settings.n_sizes),
    float(1e6)
)
ang_list = np.linspace(
    settings.ang_range_rotate[0],
    settings.ang_range_rotate[1],
    num=settings.n_ang_rotate
    # num=61
)
deg_list = np.rad2deg(ang_list)

ang_list2 = np.linspace(
    settings.ang_range_rotate[0],
    settings.ang_range_rotate[1],
    # num=settings.n_ang_rotate
    num=13
)
deg_list2 = np.rad2deg(ang_list2)

ang_list3 = np.linspace(
    settings.ang_range_rotate[0],
    settings.ang_range_rotate[1],
    # num=settings.n_ang_rotate
    num=18
)
deg_list3 = np.rad2deg(ang_list3)


conjcirchex_rot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "rotate",
    "rw",
    "circhex_rotate_60.0.pkl"
)
conjsquarehex_rot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "rotate",
    "rw",
    "sqhex_rotate_60.0.pkl"
)


clustcirchex_rot_fname = os.path.join(
    settings.loc,
    "clustering",
    "rotate",
    "rw",
    "circhex_rotate_60.0.pkl"
)
clustsquarehex_rot_fname = os.path.join(
    settings.loc,
    "clustering",
    "rotate",
    "rw",
    "sqhex_rotate_60.0.pkl"
)


repsuppcirchex_rot_fname = os.path.join(
    settings.loc,
    "repsupp",
    "rotate",
    "rw",
    "circhex_rotate_60.0.pkl"
)
repsuppsquarehex_rot_fname = os.path.join(
    settings.loc,
    "repsupp",
    "rotate",
    "rw",
    "sqhex_rotate_60.0.pkl"
)


pathcirchex_rot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "rotate",
    "rw",
    "circpathhex_rotate_60.0.pkl"
)
pathsquarehex_rot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "rotate",
    "rw",
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


conjcirchex_size_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "sizes",
    "rw",
    "circhex_sizes.pkl"
)
conjsquarehex_size_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "sizes",
    "rw",
    "sqhex_sizes.pkl"
)


clustcirchex_size_fname = os.path.join(
    settings.loc,
    "clustering",
    "sizes",
    "rw",
    "circhex_sizes.pkl"
)
clustsquarehex_size_fname = os.path.join(
    settings.loc,
    "clustering",
    "sizes",
    "rw",
    "sqhex_sizes.pkl"
)


repsuppcirchex_size_fname = os.path.join(
    settings.loc,
    "repsupp",
    "sizes",
    "rw",
    "circhex_sizes.pkl"
)
repsuppsquarehex_size_fname = os.path.join(
    settings.loc,
    "repsupp",
    "sizes",
    "rw",
    "sqhex_sizes.pkl"
)


pathcirchex_size_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "sizes",
    "rw",
    "circpathhex_sizes.pkl"
)
pathsquarehex_size_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "sizes",
    "rw",
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


###############################################################################
# Size                                                                        #
###############################################################################
size_list = np.array([30, 80, 130, 180])
offs = [0, 0]
size30_circtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot*0.6,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[0],
    sq_bound=False
)
size30_sqtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot*0.6,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[0],
    sq_bound=True
)
size80_circtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[1],
    sq_bound=False
)
size80_sqtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[1],
    sq_bound=True
)
size130_circtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot*1.5,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[2],
    sq_bound=False
)
size130_sqtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot*1.5,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[2],
    sq_bound=True
)
size180_circtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot*3,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[3],
    sq_bound=False
)
size180_sqtrajec = traj(
    dt=settings.dt,
    tmax=settings.tplot*3,
    sp=settings.speed,
    init_dir=settings.init_dir,
    dphi=settings.dphi,
    bound=size_list[3],
    sq_bound=True
)


fig = plt.figure(figsize=(16, 24))
plt.rcParams.update({'font.size': settings.fs})
spec = fig.add_gridspec(
    ncols=4, 
    nrows=6,
)


ax_circsize01 = fig.add_subplot(spec[0, 0])
get_plotting_grid(size30_circtrajec, offs, extent=200, title="Size 30 cm", titlepos = [0.5, 1.1], yticks=[-200, 0, 200], ylabel="y (cm)")
ax_circsize02 = fig.add_subplot(spec[0, 1])
get_plotting_grid(size80_circtrajec, offs, extent=200, title="Size 80 cm", titlepos = [0.5, 1.1])
ax_circsize03 = fig.add_subplot(spec[0, 2])
get_plotting_grid(size130_circtrajec, offs, extent=200, title="Size 130 cm", titlepos = [0.5, 1.1])
ax_circsize04 = fig.add_subplot(spec[0, 3])
get_plotting_grid(size180_circtrajec, offs, extent=200, title="Size 180 cm", titlepos = [0.5, 1.1])

ax_squaresize01 = fig.add_subplot(spec[1, 0])
get_plotting_grid(size30_sqtrajec, offs, extent=200, yticks=[-200, 0, 200], xticks=[0, 200], ylabel="y (cm)", xlabel="x (cm)")
ax_squaresize02 = fig.add_subplot(spec[1, 1])
get_plotting_grid(size80_sqtrajec, offs, extent=200, xticks=[-200, 0, 200], xlabel="x (cm)")
ax_squaresize03 = fig.add_subplot(spec[1, 2])
get_plotting_grid(size130_sqtrajec, offs, extent=200, xticks=[-200, 0, 200], xlabel="x (cm)")
ax_squaresize04 = fig.add_subplot(spec[1, 3])
get_plotting_grid(size180_sqtrajec, offs, extent=200, xticks=[-200, 0, 200], xlabel="x (cm)")


ax_conjsize = fig.add_subplot(spec[2, :])
ax_conjsize.spines['top'].set_visible(False)
ax_conjsize.spines['right'].set_visible(False)
plt.plot(
    sizes[:-1],
    np.mean(conjcirchex_size[:, :-1], axis=0),
    lw=2,
    label="circular boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(conjcirchex_size[:, :-1], axis=0) -
    np.std(conjcirchex_size[:, :-1], axis=0) / np.sqrt(conjcirchex_size.shape[0]),
    np.mean(conjcirchex_size[:, :-1], axis=0) +
    np.std(conjcirchex_size[:, :-1], axis=0) / np.sqrt(conjcirchex_size.shape[0]),
    alpha=0.2
)
plt.plot(
    sizes[:-1],
    np.mean(conjsquarehex_size[:, :-1], axis=0),
    lw=2,
    label="square boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(conjsquarehex_size[:, :-1], axis=0) -
    np.std(conjsquarehex_size[:, :-1], axis=0) / np.sqrt(conjsquarehex_size.shape[0]),
    np.mean(conjsquarehex_size[:, :-1], axis=0) +
    np.std(conjsquarehex_size[:, :-1], axis=0) / np.sqrt(conjsquarehex_size.shape[0]),
    alpha=0.2
)
plt.margins(0.01, 0.15)
plt.title("Conjunctive", y=0.80)
# plt.xlabel("Boundary size (cm)")
plt.ylabel("Hexasymmetry (spk/s)")
plt.xticks(np.linspace(30, 180, 6),[])
plt.yticks([59, 63, 67, 71])
plt.xlim(30, 180)
plt.ylim(59, 71)
plt.legend(prop={'size': 14})


ax_repsuppsize = fig.add_subplot(spec[3, :])
ax_repsuppsize.spines['top'].set_visible(False)
ax_repsuppsize.spines['right'].set_visible(False)
plt.plot(
    sizes[:-1],
    np.mean(repsuppcirchex_size[:, :-1], axis=0),
    lw=2,
    label="circular boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(repsuppcirchex_size[:, :-1], axis=0) -
    np.std(repsuppcirchex_size[:, :-1], axis=0) / np.sqrt(repsuppcirchex_size.shape[0]),
    np.mean(repsuppcirchex_size[:, :-1], axis=0) +
    np.std(repsuppcirchex_size[:, :-1], axis=0) / np.sqrt(repsuppcirchex_size.shape[0]),
    alpha=0.2
)
plt.plot(
    sizes[:-1],
    np.mean(repsuppsquarehex_size[:, :-1], axis=0),
    lw=2,
    label="square boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(repsuppsquarehex_size[:, :-1], axis=0) -
    np.std(repsuppsquarehex_size[:, :-1], axis=0) / np.sqrt(repsuppsquarehex_size.shape[0]),
    np.mean(repsuppsquarehex_size[:, :-1], axis=0) +
    np.std(repsuppsquarehex_size[:, :-1], axis=0) / np.sqrt(repsuppsquarehex_size.shape[0]),
    alpha=0.2
)
# plt.margins(0.01, 0.15)
plt.title("Repetition suppression", y=0.80)
plt.xticks(np.linspace(30, 180, 6),[])
plt.yticks([3, 7, 11, 15])
plt.ylabel("Hexasymmetry (spk/s)")
# # plt.xlabel("Boundary size (cm)")
# plt.ylabel("Hexasymmetry")
plt.xlim(30, 180)
plt.ylim(3, 15)


ax_clustsize = fig.add_subplot(spec[4, :])
ax_clustsize.spines['top'].set_visible(False)
ax_clustsize.spines['right'].set_visible(False)
plt.plot(
    sizes[:-1],
    np.mean(clustcirchex_size[:, :-1], axis=0),
    lw=2,
    label="circular boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(clustcirchex_size[:, :-1], axis=0) -
    np.std(clustcirchex_size[:, :-1], axis=0) / np.sqrt(clustcirchex_size.shape[0]),
    np.mean(clustcirchex_size[:, :-1], axis=0) +
    np.std(clustcirchex_size[:, :-1], axis=0) / np.sqrt(clustcirchex_size.shape[0]),
    alpha=0.2
)
plt.plot(
    sizes[:-1],
    np.mean(clustsquarehex_size[:, :-1], axis=0),
    lw=2,
    label="square boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(clustsquarehex_size[:, :-1], axis=0) -
    np.std(clustsquarehex_size[:, :-1], axis=0) / np.sqrt(clustsquarehex_size.shape[0]),
    np.mean(clustsquarehex_size[:, :-1], axis=0) +
    np.std(clustsquarehex_size[:, :-1], axis=0) / np.sqrt(clustsquarehex_size.shape[0]),
    alpha=0.2
)
# plt.margins(0.01, 0.15)
plt.title("Structure-function mapping", y=0.80)
# # plt.xlabel("Boundary size (cm)")
# plt.ylabel("Hexasymmetry")
plt.ylabel("Hexasymmetry (spk/s)")
plt.xticks(np.linspace(30, 180, 6),[])
plt.yticks([3, 7, 11, 15])
plt.xlim(30, 180)
plt.ylim(3, 15)


ax_pathsize = fig.add_subplot(spec[5, :])
ax_pathsize.spines['top'].set_visible(False)
ax_pathsize.spines['right'].set_visible(False)
plt.plot(
    sizes[:-1],
    np.mean(pathcirchex_size[:, :-1], axis=0),
    lw=2,
    label="circular boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(pathcirchex_size[:, :-1], axis=0) -
    np.std(pathcirchex_size[:, :-1], axis=0) / np.sqrt(pathcirchex_size.shape[0]),
    np.mean(pathcirchex_size[:, :-1], axis=0) +
    np.std(pathcirchex_size[:, :-1], axis=0) / np.sqrt(pathcirchex_size.shape[0]),
    alpha=0.2,
)
plt.plot(
    sizes[:-1],
    np.mean(pathsquarehex_size[:, :-1], axis=0),
    lw=2,
    label="square boundary"
)
plt.fill_between(
    sizes[:-1],
    np.mean(pathsquarehex_size[:, :-1], axis=0) -
    np.std(pathsquarehex_size[:, :-1], axis=0) / np.sqrt(pathsquarehex_size.shape[0]),
    np.mean(pathsquarehex_size[:, :-1], axis=0) +
    np.std(pathsquarehex_size[:, :-1], axis=0) / np.sqrt(pathsquarehex_size.shape[0]),
    alpha=0.2,
)
plt.xticks(np.linspace(30, 180, 6))
plt.yticks([0.003, 0.006, 0.009])
plt.xlim(30, 180)
plt.ylim(0.003, 0.009)
plt.xlabel("Boundary size (cm)")
plt.ylabel("Path\nhexasymmetry")
# plt.title("Path Hexasymmetry", y=0.80)


# plt.subplots_adjust(
#     hspace=1.1
# )

trajplot_offset = 0.003

ax_pos(ax_circsize01, -0.015, 0.025, 1., 1.)
ax_pos(ax_circsize02, -0.005, 0.025, 1., 1.)
ax_pos(ax_circsize03, 0.005, 0.025, 1., 1.)
ax_pos(ax_circsize04, 0.015, 0.025, 1., 1.)

ax_pos(ax_squaresize01, -0.015, 0.025, 1., 1.)
ax_pos(ax_squaresize02, -0.005, 0.025, 1., 1.)
ax_pos(ax_squaresize03, 0.005, 0.025, 1., 1.)
ax_pos(ax_squaresize04, 0.015, 0.025, 1., 1.)


savepath = os.path.join(
    settings.loc,
    "fig7",
    'Figure_sizes.png'
)
os.makedirs(os.path.dirname(savepath), exist_ok=True)
# plt.tight_layout()
plt.savefig(savepath)
plt.close()


###############################################################################
# Rotation                                                                    #
###############################################################################
rad_list = np.array([0, 30, 60, 90]) / 180 * np.pi
offs = [0, 0]
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


fig = plt.figure(figsize=(16, 24))
plt.rcParams.update({'font.size': settings.fs})
spec = fig.add_gridspec(
    ncols=4, 
    nrows=6,
)
# get_plotting_grid(size30_circtrajec, offs, extent=200, title="Size 30 cm", titlepos = [0.5, 1.1], yticks=[-200, 0, 200])


ax_circrot01 = fig.add_subplot(spec[0, 0])
get_plotting_grid(rot_circtrajec, offs, extent=90., yticks=[-90, 0, 90], title="Rotation of 0°", titlepos = [0.5, 1.1], ylabel="y (cm)")
ax_circrot02 = fig.add_subplot(spec[0, 1])
get_plotting_grid(rotate_traj(rot_circtrajec, rad_list[1]), offs, extent=90., title="Rotation of 30°", titlepos = [0.5, 1.1])
ax_circrot03 = fig.add_subplot(spec[0, 2])
get_plotting_grid(rotate_traj(rot_circtrajec, rad_list[2]), offs, extent=90., title="Rotation of 60°", titlepos = [0.5, 1.1])
ax_circrot04 = fig.add_subplot(spec[0, 3])
get_plotting_grid(rotate_traj(rot_circtrajec, rad_list[3]), offs, extent=90., title="Rotation of 90°", titlepos = [0.5, 1.1])

ax_squarerot01 = fig.add_subplot(spec[1, 0])
get_plotting_grid(rotate_traj(rot_sqtrajec, rad_list[0]), offs, extent=90., xticks=[-90, 0, 90], yticks=[-90, 0, 90], xlabel="x (cm)", ylabel="y (cm)")
ax_squarerot02 = fig.add_subplot(spec[1, 1])
get_plotting_grid(rotate_traj(rot_sqtrajec, rad_list[1]), offs, extent=90., xticks=[-90, 0, 90], xlabel="x (cm)")
ax_squarerot03 = fig.add_subplot(spec[1, 2])
get_plotting_grid(rotate_traj(rot_sqtrajec, rad_list[2]), offs, extent=90., xticks=[-90, 0, 90], xlabel="x (cm)")
ax_squarerot04 = fig.add_subplot(spec[1, 3])
get_plotting_grid(rotate_traj(rot_sqtrajec, rad_list[3]), offs, extent=90., xticks=[-90, 0, 90], xlabel="x (cm)")


ax_conjrot = fig.add_subplot(spec[2, :])
ax_conjrot.spines['top'].set_visible(False)
ax_conjrot.spines['right'].set_visible(False)
plt.plot(
    deg_list,
    conjcirchex_rot.mean(axis=0),
    label=f"circular boundary"
)
plt.fill_between(
    deg_list,
    conjcirchex_rot.mean(axis=0) -
    np.std(conjcirchex_rot, axis=0) / np.sqrt(conjcirchex_rot.shape[0]),
    conjcirchex_rot.mean(axis=0) +
    np.std(conjcirchex_rot, axis=0) / np.sqrt(conjcirchex_rot.shape[0]),
    alpha=0.2,
)
plt.plot(
    deg_list,
    conjsquarehex_rot.mean(axis=0),
    label=f"square boundary"
)
plt.fill_between(
    deg_list,
    conjsquarehex_rot.mean(axis=0) -
    np.std(conjsquarehex_rot, axis=0) / np.sqrt(conjsquarehex_rot.shape[0]),
    conjsquarehex_rot.mean(axis=0) +
    np.std(conjsquarehex_rot, axis=0) / np.sqrt(conjsquarehex_rot.shape[0]),
    alpha=0.2,
)
plt.title("Conjunctive", y=0.80)
# plt.xlabel("Rotation angle (degrees)")
plt.xticks(np.linspace(0, 180, 7), [])
plt.yticks(np.array([60, 65, 70]))
plt.ylabel("Hexasymmetry (spk/s)")
plt.xlim(0, 180)
plt.ylim(60, 70)
# plt.legend(prop={'size': 14})


ax_repsupprot = fig.add_subplot(spec[3, :])
ax_repsupprot.spines['top'].set_visible(False)
ax_repsupprot.spines['right'].set_visible(False)
plt.plot(
    deg_list,
    repsuppcirchex_rot.mean(axis=0),
    label=f"circle_60"
)
plt.fill_between(
    deg_list,
    repsuppcirchex_rot.mean(axis=0) -
    np.std(repsuppcirchex_rot, axis=0) / np.sqrt(repsuppcirchex_rot.shape[0]),
    repsuppcirchex_rot.mean(axis=0) +
    np.std(repsuppcirchex_rot, axis=0) / np.sqrt(repsuppcirchex_rot.shape[0]),
    alpha=0.2,
)
plt.plot(
    deg_list,
    repsuppsquarehex_rot.mean(axis=0),
    label=f"square_60"
)
plt.fill_between(
    deg_list,
    repsuppsquarehex_rot.mean(axis=0) -
    np.std(repsuppsquarehex_rot, axis=0) / np.sqrt(repsuppsquarehex_rot.shape[0]),
    repsuppsquarehex_rot.mean(axis=0) +
    np.std(repsuppsquarehex_rot, axis=0) / np.sqrt(repsuppsquarehex_rot.shape[0]),
    alpha=0.2,
)
plt.title("Repetition suppression", y=0.80)
# plt.xlabel("Rotation angle (degrees)")
plt.xticks(np.linspace(0, 180, 7), [])
plt.yticks(np.array([2, 7, 12]))
# plt.ylabel("Hexasymmetry")
plt.ylabel("Hexasymmetry (spk/s)")
plt.xlim(0, 180)
plt.ylim(2, 12)


ax_clustrot = fig.add_subplot(spec[4, :])
ax_clustrot.spines['top'].set_visible(False)
ax_clustrot.spines['right'].set_visible(False)
plt.plot(
    deg_list,
    clustcirchex_rot.mean(axis=0),
    label=f"circular boundary"
)
plt.fill_between(
    deg_list,
    clustcirchex_rot.mean(axis=0) -
    np.std(clustcirchex_rot, axis=0) / np.sqrt(clustcirchex_rot.shape[0]),
    clustcirchex_rot.mean(axis=0) +
    np.std(clustcirchex_rot, axis=0) / np.sqrt(clustcirchex_rot.shape[0]),
    alpha=0.2,
)
plt.plot(
    deg_list,
    clustsquarehex_rot.mean(axis=0),
    label=f"square boundary"
)
plt.fill_between(
    deg_list,
    clustsquarehex_rot.mean(axis=0) -
    np.std(clustsquarehex_rot, axis=0) / np.sqrt(clustsquarehex_rot.shape[0]),
    clustsquarehex_rot.mean(axis=0) +
    np.std(clustsquarehex_rot, axis=0) / np.sqrt(clustsquarehex_rot.shape[0]),
    alpha=0.2,
)
plt.title("Structure-function mapping", y=0.80)
# plt.xlabel("Rotation angle (degrees)")
plt.xticks(np.linspace(0, 180, 7), [])
plt.yticks(np.array([2, 7, 12]))
# plt.ylabel("Hexasymmetry")
plt.ylabel("Hexasymmetry (spk/s)")
plt.xlim(0, 180)
plt.ylim(2, 12)
plt.legend(prop={'size': 14})


ax_pathrot = fig.add_subplot(spec[5, :])
ax_pathrot.spines['top'].set_visible(False)
ax_pathrot.spines['right'].set_visible(False)
plt.plot(
    deg_list,
    np.mean(pathcirchex_rot, axis=0),
    lw=2,
    label="circular boundary"
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
    lw=2,
    label="square boundary"
)
plt.fill_between(
    deg_list,
    np.mean(pathsquarehex_rot, axis=0) -
    np.std(pathsquarehex_rot, axis=0) / np.sqrt(pathsquarehex_rot.shape[0]),
    np.mean(pathsquarehex_rot, axis=0) +
    np.std(pathsquarehex_rot, axis=0) / np.sqrt(pathsquarehex_rot.shape[0]),
    alpha=0.2,
)
plt.xticks(np.linspace(0, 180, 7))
plt.yticks([0.003, 0.006, 0.009])
plt.xlim(0, 180)
plt.ylim(0.003, 0.009)
plt.xlabel("Rotation angle ($^\circ$)")
plt.ylabel("Path\nhexasymmetry")
# plt.title("Path Hexasymmetry", y=0.80)


ax_pos(ax_circrot01, 0, 0.025, 1., 1.)
ax_pos(ax_circrot02, 0, 0.025, 1., 1.)
ax_pos(ax_circrot03, 0, 0.025, 1., 1.)
ax_pos(ax_circrot04, 0, 0.025, 1., 1.)

ax_pos(ax_squarerot01, 0, 0.025, 1., 1.)
ax_pos(ax_squarerot02, 0, 0.025, 1., 1.)
ax_pos(ax_squarerot03, 0, 0.025, 1., 1.)
ax_pos(ax_squarerot04, 0, 0.025, 1., 1.)


savepath = os.path.join(
    settings.loc,
    "fig7",
    'Figure_rotation.png'
)
os.makedirs(os.path.dirname(savepath), exist_ok=True)
# plt.tight_layout()
plt.savefig(savepath)
plt.close()