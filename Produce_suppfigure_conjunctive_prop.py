"""
This script simulates the conjunctive grid by head-direction cell hypothesis
for different proportions of conjunctive grid cells. This is done for each
of the three trajectory types in the manuscript. The results are plotted and
used in Figure 2-figure supplement 2.
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import settings
import matplotlib as mpl
from utils.grid_funcs import (
    gen_offsets,
    gridpop_conj,
    traj,
    traj_star,
    traj_pwl
)
from utils.utils import (
    convert_to_rhombus,
    get_hexsym,
    get_pathsym
)
from joblib import Parallel, delayed
from tqdm import tqdm

###############################################################################
# Parameters                                                                  #
###############################################################################
n_trials = 100
n_props = 100
prop_range = [0.001, 1]
props = np.logspace(
    np.log10(prop_range[0]),
    np.log10(prop_range[1]),
    n_props,
    endpoint=True
)
props = np.insert(props, 0, 0.)
print(props * 100)
fname = os.path.join(
    settings.loc,
    "conjunctive",
    "propconj",
    "data_propconj.pkl"
)
traj_types = ["star", "pwl", "rw"]


###############################################################################
# Functions                                                                   #
###############################################################################
def process_trial(i, props, traj_type):
    """
    Function to parallelize simulations over different proportions of
    conjunctive cells for each trajectory type.

    Parameters
    ----------
    i : int
        Trial number.
    props : array-like
        Proportions of conjunctive cells to simulate.
    traj_type : str
        Trajectory type to simulate. Should be "star", "pwl", or "rw"

    Returns
    -------
    trial_data : dict
        Dictionary containing firing rates, hexasymmetry and path
        hexasymmetry for each proportion of conjunctive cells.
    """
    assert traj_type in ["star", "pwl", "rw"], \
        "traj_type must be 'star', 'pwl' or 'rw'"
    trial_data = {
        'i_fr': np.zeros(len(props)),
        'i_hexes': np.zeros(len(props)),
        'i_paths': np.zeros(len(props)),
        'r_fr': np.zeros(len(props)),
        'r_hexes': np.zeros(len(props)),
        'r_paths': np.zeros(len(props))
    }

    for j, prop in enumerate(props):
        # Generate offsets for conjunctive grid cells
        ox, oy = gen_offsets(N=settings.N, kappacl=0.)
        oxr, oyr = convert_to_rhombus(ox, oy)

        # Generate trajectory
        if traj_type == "star":
            trajec = traj_star(settings.phbins, settings.rmax, settings.dt)
        if traj_type == "pwl":
            trajec = traj_pwl(settings.phbins, settings.rmax, settings.dt)
        if traj_type == "rw":
            trajec = traj(settings.dt, settings.tmax, settings.speed)

        # Firing rates, hexasymmetry, and path hexasymmetry for ideal params
        _, mean_fr, _, summed_fr = gridpop_conj(
            settings.N,
            settings.grsc,
            settings.phbins,
            traj=trajec,
            oxr=oxr,
            oyr=oyr,
            propconj=prop,
            kappa=settings.kappac_i,
            jitter=settings.jitterc_i
        )
        trial_data['i_fr'][j] = np.mean(mean_fr)
        trial_data['i_hexes'][j] = get_hexsym(summed_fr, trajec)
        trial_data['i_paths'][j] = get_pathsym(trajec)

        # Firing rates, hexasymmetry, and path hexasymmetry for real params
        _, mean_fr, _, summed_fr = gridpop_conj(
            settings.N,
            settings.grsc,
            settings.phbins,
            traj=trajec,
            oxr=oxr,
            oyr=oyr,
            propconj=prop,
            kappa=settings.kappac_r,
            jitter=settings.jitterc_r
        )
        trial_data['r_fr'][j] = np.mean(mean_fr)
        trial_data['r_hexes'][j] = get_hexsym(summed_fr, trajec)
        trial_data['r_paths'][j] = get_pathsym(trajec)

    return trial_data


###############################################################################
# Run                                                                         #
###############################################################################
if not os.path.isfile(fname):
    data = {
        'star': {
            'ideal': {'fr': [], 'hexes': [], 'paths': []},
            'real': {'fr': [], 'hexes': [], 'paths': []}
        },
        'pwl': {
            'ideal': {'fr': [], 'hexes': [], 'paths': []},
            'real': {'fr': [], 'hexes': [], 'paths': []}
        },
        'rw': {
            'ideal': {'fr': [], 'hexes': [], 'paths': []},
            'real': {'fr': [], 'hexes': [], 'paths': []}
        }
    }

    for traj_type in traj_types:
        print(traj_type)
        results = Parallel(n_jobs=50)(
            delayed(
                process_trial
            )(i, props, traj_type) for i in tqdm(range(n_trials))
        )

        # Combine results
        for result in results:
            data[traj_type]['ideal']['fr'].append(result['i_fr'])
            data[traj_type]['ideal']['hexes'].append(result['i_hexes'])
            data[traj_type]['ideal']['paths'].append(result['i_paths'])
            data[traj_type]['real']['fr'].append(result['r_fr'])
            data[traj_type]['real']['hexes'].append(result['r_hexes'])
            data[traj_type]['real']['paths'].append(result['r_paths'])

        # Convert lists to numpy arrays
        for key in data[traj_type]['ideal']:
            data[traj_type]['ideal'][key] = np.array(
                data[traj_type]['ideal'][key]
            )
            data[traj_type]['real'][key] = np.array(
                data[traj_type]['real'][key]
            )

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
else:
    with open(fname, 'rb') as f:
        data = pickle.load(f)

###############################################################################
# Plotting                                                                    #
###############################################################################
real_id = np.argmin(np.abs(props - 0.33))   # index of realistic proportion

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': settings.fs})
plt.figure(figsize=(8, 5))
for traj_type in traj_types:
    # Plot median hexasymmetry
    plt.plot(
        props,
        np.median(data[traj_type]["real"]["hexes"], axis=0),
        label=f"{traj_type}",
        zorder=10
    )
    # Plot standard error of hexasymmetry
    plt.fill_between(
        props,
        np.median(data[traj_type]["real"]["hexes"], axis=0) -
        np.std(
            data[traj_type]["real"]["hexes"], axis=0
        ) / np.sqrt(data[traj_type]["real"]["hexes"].shape[0]),
        np.median(data[traj_type]["real"]["hexes"], axis=0) +
        np.std(
            data[traj_type]["real"]["hexes"], axis=0
        ) / np.sqrt(data[traj_type]["real"]["hexes"].shape[0]),
        alpha=0.8,
        zorder=10
    )
plt.plot(
    props[30:],
    # 17 is an arbitrarily chosen num to show linear dependence for large prop
    props[30:] * 17,
    linestyle="--",
    linewidth=2.,
    color="black",
    zorder=0
)
plt.gca().set_aspect('equal')
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Hexasymmetry (spk/s)")
plt.xlabel("Proportion of conjunctive cells")
plt.legend()
plt.tight_layout()

plot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "propconj",
    "figure_propconj.png"
)
os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(plot_fname, dpi=300)
