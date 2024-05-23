"""
This script calculates the hexasymmetry when using trajectories from rat
trajectories (Sargolini et al., 2006) and human trajectories (Kunz et al. 2015,
2021). All three hypotheses (conjunctive grid by head-direction cell
hypothesis, repetition-suppression hypothesis, and structure-function mapping
hypothesis) are considered. The hexasymmetries are then plotted for comparison.
One example trajectory from each dataset is also plotted overlayed on the
firing field of one grid cell for illustration purposes. The resulting plot is
used in Supplementary Figure S9 of the manuscript.
"""
import csv
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import settings
from utils.grid_funcs import (
    gen_offsets,
    gridpop_clustering,
    gridpop_conj,
    gridpop_repsupp,
    load_traj
)
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from utils.utils import (
    convert_to_rhombus, get_hexsym, get_pathsym, grid_meanfr
)


###############################################################################
# Functions                                                                   #
###############################################################################
def get_unique_filenames(directory):
    """
    Finds all unique filenames in a given directory.
    """
    filenames = os.listdir(directory)
    unique_prefixes = []

    for filename in filenames:
        prefix = filename.split("_")[0]
        if prefix not in unique_prefixes:
            unique_prefixes.append(prefix)

    return unique_prefixes


def load_csv2numpy(csv_file_path):
    """
    Loads a .csv file and converts its contents into a numpy array.
    """
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        data = np.array(list(reader)[0]).astype(float)
        data = data[~np.isnan(data)]

    return data


def compute_movdirs(x, y):
    """
    Calculates the vector of headings for a trajectory defined by the vectors x
    and y. The heading at the zeroth time step is randomly chosen in the
    interval [0, 2pi). Returns the vector of headings.
    """
    # Compute the difference between successive co-ordinates
    dx = np.diff(x)
    dy = np.diff(y)

    # Compute the direction in each timestep using np.arctan2 dy and dx which
    # is between -pi and pi
    direction = np.arctan2(dy, dx)

    # To handle the zeroth time step, we insert a random value at the start in
    # range [-pi, pi]
    zeroth_time_step_direction = 2 * np.pi * random.random() - np.pi
    direction = np.insert(direction, 0, zeroth_time_step_direction)

    return direction


def clean_traj(traj) -> np.ndarray:
    """
    For a given trajectory, removes coordinates where the subject has not
    moved. The list 'traj' should have the structure [t, x, y, direc] where
    't', 'x', 'y', and 'direc' are numpy.ndarrays. Returns the cleaned
    trajectory.
    """
    # Truncate the elements
    min_length = min(len(lst) for lst in traj)
    traj = [lst[:min_length] for lst in traj]

    traj = np.array(traj)
    diffd = np.diff(traj, axis=0)[:, 1:3]
    # Ensure all elements in diffd have the same shape as the comparison array
    comparison_array = np.zeros(diffd.shape)

    # Check if all differences are zero
    condition = np.all(diffd == comparison_array, axis=1)

    if np.any(condition):
        idx = np.where(condition)[0] + 1
        traj_cleaned = np.delete(traj, idx, axis=0)
    else:
        traj_cleaned = traj

    return traj_cleaned


def compute_hexes(n, id, r, ox_rand, oy_rand):
    trajec = np.array([r[:, 0], r[:, 1], r[:, 2], r[:, 3]])

    oxr_rand, oyr_rand = convert_to_rhombus(ox_rand, oy_rand)

    gridpop_params = {
        "N": settings.N,
        "grsc": settings.grsc,
        "phbins": settings.phbins,
        "traj": trajec,
        "oxr": oxr_rand,
        "oyr": oyr_rand,
    }

    pathsym = get_pathsym(trajec)

    meanfr, summed_fr = gridpop_conj(**gridpop_params)[1::2]
    hex_conj = get_hexsym(summed_fr, trajec)
    path_conj = pathsym * np.mean(meanfr)

    meanfr, summed_fr = gridpop_repsupp(**gridpop_params)[1::2]
    hex_repsupp = get_hexsym(summed_fr, trajec)
    path_repsupp = pathsym * np.mean(meanfr)

    meanfr, summed_fr = gridpop_clustering(**gridpop_params)[1::2]
    hex_clust = get_hexsym(summed_fr, trajec)
    path_clust = pathsym * np.mean(meanfr)

    hexes = np.array(
        [hex_conj, hex_repsupp, hex_clust, path_conj, path_repsupp, path_clust]
    )

    return hexes


###############################################################################
# Parameters                                                                  #
###############################################################################
# directory to load rat trajectories from
rat_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "Data_sargolini_csv_more"
    )
)

# filename to save rat results to
fname_rats = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "Data_sargolini_csv_more",
        "hexes.pkl"
    )
)

# filenames to save human  results to
fname_th = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "paths",
        "hexes_th.pkl"
    )
)
fname_of = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "paths",
        "hexes_of.pkl"
    )
)


###############################################################################
# Simulation                                                                  #
###############################################################################
# simulations on rat trajectories
if not os.path.isfile(fname_rats):
    trial_names = get_unique_filenames(rat_dir)
    hexes_rats = np.array([])

    for trial_name in trial_names:
        t_name = os.path.join(
            rat_dir,
            trial_name + "_post.csv"
        )
        x_name = os.path.join(
            rat_dir,
            trial_name + "_posx.csv"
        )
        y_name = os.path.join(
            rat_dir,
            trial_name + "_posy.csv"
        )

        t = load_csv2numpy(t_name)
        x = load_csv2numpy(x_name)
        y = load_csv2numpy(y_name)
        direc = compute_movdirs(x, y)

        traj = [t, x, y, direc]
        traj = clean_traj(traj)

        ox_rand, oy_rand = gen_offsets(N=settings.N, kappacl=0.)
        oxr_rand, oyr_rand = convert_to_rhombus(ox_rand, oy_rand)

        gridpop_params = {
            "N": settings.N,
            "grsc": settings.grsc,
            "phbins": settings.phbins,
            "traj": traj,
            "oxr": oxr_rand,
            "oyr": oyr_rand,
        }

        pathsym = get_pathsym(traj)

        try:
            meanfr, summed_fr = gridpop_conj(**gridpop_params)[1::2]
            hex_conj = get_hexsym(summed_fr, traj)
            path_conj = pathsym * np.mean(meanfr)
        except ValueError:
            hex_conj = np.nan
            path_conj = np.nan

        try:
            meanfr, summed_fr = gridpop_repsupp(**gridpop_params)[1::2]
            hex_repsupp = get_hexsym(summed_fr, traj)
            path_repsupp = pathsym * np.mean(meanfr)
        except ValueError:
            hex_repsupp = np.nan
            path_repsupp = np.nan

        try:
            meanfr, summed_fr = gridpop_clustering(**gridpop_params)[1::2]
            hex_clust = get_hexsym(summed_fr, traj)
            path_clust = pathsym * np.mean(meanfr)
        except ValueError:
            hex_clust = np.nan
            path_clust = np.nan

        if hexes_rats.size:
            hexes_rats = np.vstack(
                (
                    hexes_rats,
                    np.array(
                        [
                            hex_conj,
                            hex_repsupp,
                            hex_clust,
                            path_conj,
                            path_repsupp,
                            path_clust
                        ]
                    )
                )
            )
        else:
            hexes_rats = np.array(
                [
                    hex_conj,
                    hex_repsupp,
                    hex_clust,
                    path_conj,
                    path_repsupp,
                    path_clust
                ]
            )
        print(hexes_rats)

    os.makedirs(os.path.dirname(fname_rats), exist_ok=True)
    with open(fname_rats, 'wb') as f:
        pickle.dump(hexes_rats, f)
else:
    with open(fname_rats, 'rb') as f:
        hexes_rats = pickle.load(f)

# simulations on human 'TH' trajectories
if not os.path.isfile(fname_th):
    hexes_th = np.array([])
    trajecs_th = load_traj()[0]
    # print(trajecs)

    ox_rand, oy_rand = gen_offsets(N=settings.N, kappacl=0.)

    hexes_th = Parallel(n_jobs=-1)(
        delayed(
            compute_hexes
        )(
            n, id, r, ox_rand, oy_rand
        ) for n, (id, r) in tqdm(enumerate(trajecs_th.items()))
    )
    hexes_th = np.vstack(hexes_th)

    os.makedirs(os.path.dirname(fname_th), exist_ok=True)
    with open(fname_th, 'wb') as f:
        pickle.dump(hexes_th, f)
else:
    with open(fname_th, 'rb') as f:
        hexes_th = pickle.load(f)

# simulations on human 'OF' trajectories
if not os.path.isfile(fname_of):
    hexes_of = np.array([])
    trajecs_of = load_traj()[1]

    ox_rand, oy_rand = gen_offsets(N=settings.N, kappacl=0.)

    hexes_of = Parallel(n_jobs=-1)(
        delayed(
            compute_hexes
        )(
            n, id, r, ox_rand, oy_rand
        ) for n, (id, r) in tqdm(enumerate(trajecs_of.items()))
    )
    hexes_of = np.vstack(hexes_of)

    os.makedirs(os.path.dirname(fname_of), exist_ok=True)
    with open(fname_of, 'wb') as f:
        pickle.dump(hexes_of, f)
else:
    with open(fname_of, 'rb') as f:
        hexes_of = pickle.load(f)

###############################################################################
# Plotting                                                                    #
###############################################################################
hexes_col = [
    "conj_h", "repsupp_h", "clust_h", "conj_m", "repsupp_m", "clust_m"
]


def create_dict(dataset):
    return {
        k: dataset[:, i][~np.isnan(dataset[:, i])] for i, k in enumerate(
            hexes_col
        )
    }


def calculate_significance(data1, data2):
    """
    Performs a Mann-Whitney U test between datasets data1 and data2, then
    returns a string denoting the significance from the calculated p value.
    """
    u, p = mannwhitneyu(data1, data2)
    if p < 0.001:
        significance = "***"
    elif p < 0.01:
        significance = "**"
    elif p < 0.05:
        significance = "*"
    else:
        significance = "n.s."

    return significance


def set_violin_color(violins, color):
    for pc in violins['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    violins["cbars"].set_color(color)
    violins["cmins"].set_color(color)
    violins["cmaxes"].set_color(color)


def do_violin_plots(data_dict, positions, colors, xoff=0.2):
    for i, color in enumerate(colors):
        violins = plt.violinplot(
            [data_dict[key] for key in hexes_col[i*3:i*3+3]],
            positions + xoff*(i - 0.5),
            widths=0.35
        )
        set_violin_color(violins, color)


def bar_chart(pos, data_dict, colors):
    for i, key in enumerate(data_dict):
        plt.bar(pos[i % 3], np.mean(data_dict[key]), color=colors[i])


def significance_markers(pos, data_dict1, data_dict2):
    for i in range(3):
        data1 = data_dict1[hexes_col[i]]
        data2 = data_dict2[hexes_col[i+3]]
        plt.text(
            i + pos,
            max(data1+data2),
            calculate_significance(data1, data2),
            ha='center'
        )


rathexes_dict = create_dict(hexes_rats)
thhexes_dict = create_dict(hexes_th)
ofhexes_dict = create_dict(hexes_of)

bar_col = ['b']*3 + [[0.67, 0.85, 0.9]]*3
labels1 = [
    'Conjunctive', 'Repetition-\nsuppression', 'Structure-\nfunction\nmapping'
]
labels2 = [
    'Conjunctive', 'Repetition-\nsuppression', 'Structure-\nfunction\nmapping'
]
labels3 = [
    'Conjunctive', 'Repetition-\nsuppression', 'Structure-\nfunction\nmapping'
]

pos1, pos2, pos3 = np.arange(3), np.arange(3)+3.5, np.arange(3)+7.0

# plt.figure(figsize=(18, 8))
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, height_ratios=[8, 4])
plt.rcParams.update({'font.size': settings.fs})

ax_main = fig.add_subplot(gs[0, :])
ax_main.set_yscale('log')
bar_chart(pos1, rathexes_dict, bar_col)
bar_chart(pos2, thhexes_dict, bar_col)
bar_chart(pos3, ofhexes_dict, bar_col)

plt.xticks(
    np.concatenate((pos1, pos2, pos3)),
    np.concatenate((labels1, labels2, labels3)),
    fontsize=settings.fs*0.8
)
plt.ylabel('Hexasymmetry (spk/s)')

do_violin_plots(rathexes_dict, pos1, ['red', 'orange'])
do_violin_plots(thhexes_dict, pos2, ['red', 'orange'])
do_violin_plots(ofhexes_dict, pos3, ['red', 'orange'])

plt.vlines((2 + 3.5)/2, 1e-1, 1e4, color="black", linewidth=1.)
plt.vlines((5.5 + 7.0)/2, 1e-1, 1e4, color="black", linewidth=1.)

significance_markers(0, rathexes_dict, rathexes_dict)
significance_markers(3.5, thhexes_dict, thhexes_dict)
significance_markers(7.0, ofhexes_dict, ofhexes_dict)

ax_main.text(
    0.18,
    0.95,
    "Rat trajectories " +
    f"({len(rathexes_dict[list(rathexes_dict.keys())[0]])} trials)",
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax_main.transAxes
)
ax_main.text(
    0.5,
    0.95,
    "Human trajectories (TH) " +
    f"({len(thhexes_dict[list(thhexes_dict.keys())[0]])} trials)",
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax_main.transAxes
)
ax_main.text(
    0.83,
    0.95,
    "Human trajectories (OF) " +
    f"({len(ofhexes_dict[list(thhexes_dict.keys())[0]])} trials)",
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax_main.transAxes
)

ax_main.set_ylim([1e-1, 1e4])

trial_names = get_unique_filenames(rat_dir)

idx = np.random.randint(len(trial_names))
trial_name = trial_names[idx]
t_name = os.path.join(
    rat_dir,
    trial_name + "_post.csv"
)
x_name = os.path.join(
    rat_dir,
    trial_name + "_posx.csv"
)
y_name = os.path.join(
    rat_dir,
    trial_name + "_posy.csv"
)

t = load_csv2numpy(t_name)
x = load_csv2numpy(x_name)
y = load_csv2numpy(y_name)
direc = compute_movdirs(x, y)

traj_rat = [t, x, y, direc]
traj_rat = clean_traj(traj_rat)

trajecs_th = load_traj()[0]
trajecs_of = load_traj()[1]

r_th = trajecs_th[random.choice(list(trajecs_th.keys()))]
r_of = trajecs_of[random.choice(list(trajecs_of.keys()))]
trajec_th = np.array([r_th[:, 0], r_th[:, 1], r_th[:, 2], r_th[:, 3]])
trajec_of = np.array([r_of[:, 0], r_of[:, 1], r_of[:, 2], r_of[:, 3]])


def prepare_grid(
        x_min,
        x_max,
        y_min,
        y_max,
        count,
        grsc=30.,
        angle=0.,
        offs=np.array([0, 0])
):
    X_bgr, Y_bgr, _ = np.meshgrid(
        np.linspace(x_min, x_max, count),
        np.linspace(y_min, y_max, count),
        1
    )
    gr_bgr = grid_meanfr(X_bgr, Y_bgr, grsc=grsc, angle=angle, offs=offs)
    return X_bgr, Y_bgr, gr_bgr


def overlay_trajongrid(ax, x, y, X_bgr, Y_bgr, gr_bgr):
    """
    On a given matplotlib figure axis ax, plots the trajectory defined by x and
    y overlayed ontomthe 2d sheet of grid cell firing fields of magnitude
    gr_bgr defined on the grid X_bgr and Y_bgr.

    Parameters
    ----------
        ax (Axes):              matplotlib axis
        x (np.ndarray):         vector of a trajectory's x coordinates
        y (np.ndarray):         vector of a trajectory's y coordinates
        X_bgr (np.ndarray):     2d array of grid x coordinates
        Y_bgr (np.ndarray):     2d array of grid y coordinates
        gr_bgr (np.ndarray):    2d array of grid cell firing field values
    """
    ax.plot(x, y, color="red")
    ax.pcolormesh(
        X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
    )
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_xticks([])
    ax.set_yticks([])


X_bgr, Y_bgr, gr_bgr = prepare_grid(-200, 200, -200, 200, 600)

ax_traj_rat = fig.add_subplot(gs[1, 0], aspect='equal')
overlay_trajongrid(ax_traj_rat, x, y, X_bgr, Y_bgr, gr_bgr)

ax_traj_th = fig.add_subplot(gs[1, 1], aspect='equal')
x, y = trajec_th[1], trajec_th[2]
overlay_trajongrid(ax_traj_th, x, y, X_bgr, Y_bgr, gr_bgr)


ax_traj_of = fig.add_subplot(gs[1, 2], aspect='equal')
x, y = trajec_of[1], trajec_of[2]
x, y = x[:int(len(x))], y[:int(len(x))]
overlay_trajongrid(ax_traj_of, x, y, X_bgr, Y_bgr, gr_bgr)


plt.tight_layout()
plt.show()
plt.close()
