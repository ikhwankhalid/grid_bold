import numpy as np
from joblib import Parallel, delayed
import settings
from utils.grid_funcs import (
    traj,
    gen_offsets,
    traj_pwl,
    traj_star,
    gridpop_clustering,
    gridpop_const
)
from utils.utils import (
    convert_to_rhombus,
    get_hexsym as get_hexsym2,
    get_pathsym
)
import os
import re
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

###############################################################################
# Parameters & array initialisation                                           #
###############################################################################
rep = settings.rep

h_cl_star = np.zeros((3, rep))
h_cl_rw = np.zeros((3, rep))
h_cl_pl = np.zeros((3, rep))


###############################################################################
# Functions                                                                   #
###############################################################################
def mfunc(i, oxr, oyr):
    ################
    # random walks #
    ################
    trajec = traj(settings.dt, settings.tmax)

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_rw = (
        get_hexsym2(summed_fr, trajec),
        get_pathsym(trajec),
        np.mean(fr_mean)
    )

    ###################
    # star-like walks #
    ###################
    star_offs = np.random.rand(2)
    trajec_star = traj_star(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed,
        offset=star_offs
    )

    # clustering
    direc_binned, meanfr, _, summed_fr = gridpop_clustering(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_star = (
        get_hexsym2(summed_fr, trajec_star),
        get_pathsym(trajec_star),
        np.mean(meanfr[~np.isnan(meanfr).any()])
    )

    #####################
    # piece-wise linear #
    #####################
    trajec_pl = traj_pwl(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_pl,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_pl = (
        get_hexsym2(summed_fr, trajec_pl),
        get_pathsym(trajec_pl),
        np.mean(fr_mean)
    )

    return (
        h_cl_star,
        h_cl_rw,
        h_cl_pl,
    )


def mfunc_const(i, oxr, oyr):
    ################
    # random walks #
    ################
    trajec = traj(settings.dt, settings.tmax)

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_const(
        N=settings.N,
        phbins=settings.phbins,
        traj=trajec
    )
    h_cl_rw = (
        get_hexsym2(summed_fr, trajec),
        get_pathsym(trajec),
        np.mean(fr_mean)
    )

    ###################
    # star-like walks #
    ###################
    star_offs = np.random.rand(2)
    trajec_star = traj_star(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed,
        offset=star_offs
    )

    # clustering
    direc_binned, meanfr, _, summed_fr = gridpop_const(
        settings.N,
        phbins=settings.phbins,
        traj=trajec_star
    )
    h_cl_star = (
        get_hexsym2(summed_fr, trajec_star),
        get_pathsym(trajec_star),
        np.mean(meanfr[~np.isnan(meanfr).any()])
    )

    #####################
    # piece-wise linear #
    #####################
    trajec_pl = traj_pwl(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_const(
        N=settings.N,
        phbins=settings.phbins,
        traj=trajec_pl
    )
    h_cl_pl = (
        get_hexsym2(summed_fr, trajec_pl),
        get_pathsym(trajec_pl),
        np.mean(fr_mean)
    )

    return (
        h_cl_star,
        h_cl_rw,
        h_cl_pl,
    )


###############################################################################
# Run                                                                         #
###############################################################################
ox, oy = gen_offsets(N=settings.N, kappacl=0.)
oxr, oyr = convert_to_rhombus(ox, oy)

if False:
    for i in range(5):
        print("sim: ", i)
        alldata = Parallel(
            n_jobs=-1, verbose=100
        )(delayed(mfunc)(i, oxr, oyr) for i in tqdm(range(rep)))
        alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)

        (
            h_cl_star,
            h_cl_rw,
            h_cl_pl
        ) = alldata

        # same figure for realistic parameter choices
        data = np.array(
            [
                h_cl_star[0, :], h_cl_star[1, :], h_cl_star[2, :],
                h_cl_pl[0, :], h_cl_pl[1, :], h_cl_pl[2, :],
                h_cl_rw[0, :], h_cl_rw[1, :], h_cl_rw[2, :]
            ]
        )

        # Define the directory name
        dir_name = os.path.join(settings.loc, "clustering", "no-clustering")
        # Check if the directory exists
        if not os.path.exists(dir_name):
            # If not, create it
            os.makedirs(dir_name, exist_ok=True)

        files = os.listdir(dir_name)

        # Extract their sequence numbers
        sequence_numbers = [re.findall(r'_(\d+)-(\d+)', file) for file in files]
        # Flatten the list
        sequence_numbers = [
            item for sublist in sequence_numbers for item in sublist]
        sequence_numbers = [tuple(map(int, pair))
                            for pair in sequence_numbers if pair]

        # Get the highest sequence number present
        if sequence_numbers:
            rep_low = max([pair[1] for pair in sequence_numbers]) + 1
            rep_high = rep_low + rep - 1
        else:
            rep_high = rep
            rep_low = 1

        print(rep_low, rep_high)

        # Define the full file name
        filename = os.path.join(
            dir_name,
            f"no-clustering_{rep_low}-{rep_high}"
        )

        # Save the data
        np.save(filename, data)

ox2, oy2 = np.meshgrid(
    np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False),
    np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False)
)
ox2r = ox2.reshape(1, -1)[0]
oy2r = oy2.reshape(1, -1)[0]
ox2r, oy2r = convert_to_rhombus(ox2r, oy2r)

if False:
    for i in range(5):
        print("sim: ", i)
        alldata = Parallel(
            n_jobs=-1, verbose=100
        )(delayed(mfunc)(i, ox2r, oy2r) for i in tqdm(range(rep)))
        alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)

        (
            h_cl_star,
            h_cl_rw,
            h_cl_pl
        ) = alldata

        # same figure for realistic parameter choices
        data = np.array(
            [
                h_cl_star[0, :], h_cl_star[1, :], h_cl_star[2, :],
                h_cl_pl[0, :], h_cl_pl[1, :], h_cl_pl[2, :],
                h_cl_rw[0, :], h_cl_rw[1, :], h_cl_rw[2, :]
            ]
        )

        # Define the directory name
        dir_name = os.path.join(
            settings.loc, "clustering", "no-clustering-uniform"
        )
        # Check if the directory exists
        if not os.path.exists(dir_name):
            # If not, create it
            os.makedirs(dir_name, exist_ok=True)

        files = os.listdir(dir_name)

        # Extract their sequence numbers
        sequence_numbers = [re.findall(r'_(\d+)-(\d+)', file) for file in files]
        # Flatten the list
        sequence_numbers = [
            item for sublist in sequence_numbers for item in sublist]
        sequence_numbers = [tuple(map(int, pair))
                            for pair in sequence_numbers if pair]

        # Get the highest sequence number present
        if sequence_numbers:
            rep_low = max([pair[1] for pair in sequence_numbers]) + 1
            rep_high = rep_low + rep - 1
        else:
            rep_high = rep
            rep_low = 1

        print(rep_low, rep_high)

        # Define the full file name
        filename = os.path.join(
            dir_name,
            f"no-clustering_{rep_low}-{rep_high}"
        )

        # Save the data
        np.save(filename, data)

ox2, oy2 = np.meshgrid(
    np.linspace(0, 1, int(np.sqrt(64)), endpoint=False),
    np.linspace(0, 1, int(np.sqrt(64)), endpoint=False)
)
ox2r = ox2.reshape(1, -1)[0]
oy2r = oy2.reshape(1, -1)[0]
ox2r, oy2r = convert_to_rhombus(ox2r, oy2r)

if False:
    for i in range(5):
        print("sim: ", i)
        alldata = Parallel(
            n_jobs=-1, verbose=100
        )(delayed(mfunc)(i, ox2r, oy2r) for i in tqdm(range(rep)))
        alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)

        (
            h_cl_star,
            h_cl_rw,
            h_cl_pl
        ) = alldata

        # same figure for realistic parameter choices
        data = np.array(
            [
                h_cl_star[0, :], h_cl_star[1, :], h_cl_star[2, :],
                h_cl_pl[0, :], h_cl_pl[1, :], h_cl_pl[2, :],
                h_cl_rw[0, :], h_cl_rw[1, :], h_cl_rw[2, :]
            ]
        )

        # Define the directory name
        dir_name = os.path.join(
            settings.loc, "clustering", "no-clustering-uniform-256"
        )
        # Check if the directory exists
        if not os.path.exists(dir_name):
            # If not, create it
            os.makedirs(dir_name, exist_ok=True)

        files = os.listdir(dir_name)

        # Extract their sequence numbers
        sequence_numbers = [re.findall(r'_(\d+)-(\d+)', file) for file in files]
        # Flatten the list
        sequence_numbers = [
            item for sublist in sequence_numbers for item in sublist]
        sequence_numbers = [tuple(map(int, pair))
                            for pair in sequence_numbers if pair]

        # Get the highest sequence number present
        if sequence_numbers:
            rep_low = max([pair[1] for pair in sequence_numbers]) + 1
            rep_high = rep_low + rep - 1
        else:
            rep_high = rep
            rep_low = 1

        print(rep_low, rep_high)

        # Define the full file name
        filename = os.path.join(
            dir_name,
            f"no-clustering_{rep_low}-{rep_high}"
        )

        # Save the data
        np.save(filename, data)

ox, oy = gen_offsets(N=settings.N, kappacl=0.)
oxr, oyr = convert_to_rhombus(ox, oy)

if False:
    for i in range(5):
        print("sim: ", i)
        alldata = Parallel(
            n_jobs=-1, verbose=100
        )(delayed(mfunc_const)(i, oxr, oyr) for i in tqdm(range(rep)))
        alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)

        (
            h_cl_star,
            h_cl_rw,
            h_cl_pl
        ) = alldata

        # same figure for realistic parameter choices
        data = np.array(
            [
                h_cl_star[0, :], h_cl_star[1, :], h_cl_star[2, :],
                h_cl_pl[0, :], h_cl_pl[1, :], h_cl_pl[2, :],
                h_cl_rw[0, :], h_cl_rw[1, :], h_cl_rw[2, :]
            ]
        )

        # Define the directory name
        dir_name = os.path.join(
            settings.loc, "clustering", "no-clustering-const"
        )
        # Check if the directory exists
        if not os.path.exists(dir_name):
            # If not, create it
            os.makedirs(dir_name, exist_ok=True)

        files = os.listdir(dir_name)

        # Extract their sequence numbers
        sequence_numbers = [re.findall(r'_(\d+)-(\d+)', file) for file in files]
        # Flatten the list
        sequence_numbers = [
            item for sublist in sequence_numbers for item in sublist]
        sequence_numbers = [tuple(map(int, pair))
                            for pair in sequence_numbers if pair]

        # Get the highest sequence number present
        if sequence_numbers:
            rep_low = max([pair[1] for pair in sequence_numbers]) + 1
            rep_high = rep_low + rep - 1
        else:
            rep_high = rep
            rep_low = 1

        print(rep_low, rep_high)

        # Define the full file name
        filename = os.path.join(
            dir_name,
            f"no-clustering_{rep_low}-{rep_high}"
        )

        # Save the data
        np.save(filename, data)


###############################################################################
# Plotting                                                                    #
###############################################################################
dir_name = os.path.join(settings.loc, "clustering", "no-clustering")
dir_uniform_name = os.path.join(
    settings.loc, "clustering", "no-clustering-uniform"
)
dir_const_name = os.path.join(
    settings.loc, "clustering", "no-clustering-const"
)
total = 300
conditions = 3

a6_0kappa = np.zeros((total, conditions))
m6_0kappa = np.zeros((total, conditions))
a0_0kappa = np.zeros((total, conditions))
a6_uniform = np.zeros((total, conditions))
m6_uniform = np.zeros((total, conditions))
a0_uniform = np.zeros((total, conditions))
a6_const = np.zeros((total, conditions))
m6_const = np.zeros((total, conditions))
a0_const = np.zeros((total, conditions))
a6_real = np.zeros((total, 10))
m6_real = np.zeros((total, 10))
a0_real = np.zeros((total, 10))
for i in np.arange(0, total, settings.rep):
    temp_0kappa = np.load(
        os.path.join(
            dir_name,
            "no-clustering_" +
            str(i+1) + '-' + str(i + settings.rep) + '.npy'
        ),
        allow_pickle=True
    )
    temp_real = np.load(
        os.path.join(
            settings.loc,
            "summary",
            "real",
            "summary_bar_plot_3hyp3traj_" +
            str(i+1) + '-' + str(i + settings.rep) + '.npy'
        ),
        allow_pickle=True
    )
    temp_uniform = np.load(
        os.path.join(
            dir_uniform_name,
            "no-clustering_" +
            str(i+1) + '-' + str(i + settings.rep) + '.npy'
        ),
        allow_pickle=True
    )
    temp_const = np.load(
        os.path.join(
            dir_const_name,
            "no-clustering_" +
            str(i+1) + '-' + str(i + settings.rep) + '.npy'
        ),
        allow_pickle=True
    )
    temp_0kappa = np.reshape(temp_0kappa, (conditions, 3, settings.rep))
    temp_real = np.reshape(temp_real, (10, 3, settings.rep))
    temp_uniform = np.reshape(temp_uniform, (conditions, 3, settings.rep))
    temp_const = np.reshape(temp_const, (conditions, 3, settings.rep))

    a6_0kappa[i:(i+settings.rep), :] = temp_0kappa[:, 0, :].T
    m6_0kappa[i:(i+settings.rep), :] = temp_0kappa[:, 1, :].T
    a0_0kappa[i:(i+settings.rep), :] = temp_0kappa[:, 2, :].T
    a6_real[i:(i+settings.rep), :] = temp_real[:, 0, :].T
    m6_real[i:(i+settings.rep), :] = temp_real[:, 1, :].T
    a0_real[i:(i+settings.rep), :] = temp_real[:, 2, :].T
    a6_uniform[i:(i+settings.rep), :] = temp_uniform[:, 0, :].T
    m6_uniform[i:(i+settings.rep), :] = temp_uniform[:, 1, :].T
    a0_uniform[i:(i+settings.rep), :] = temp_uniform[:, 2, :].T
    a6_const[i:(i+settings.rep), :] = temp_const[:, 0, :].T
    m6_const[i:(i+settings.rep), :] = temp_const[:, 1, :].T
    a0_const[i:(i+settings.rep), :] = temp_const[:, 2, :].T

a6_real[:, 3:9] = np.roll(a6_real[:, 3:9], 3, 1)
m6_real[:, 3:9] = np.roll(m6_real[:, 3:9], 3, 1)
a0_real[:, 3:9] = np.roll(a0_real[:, 3:9], 3, 1)

hexes = [
    a6_0kappa[:, 0],
    a6_0kappa[:, 1],
    a6_0kappa[:, 2],
    a6_real[:, 6],
    a6_real[:, 7],
    a6_real[:, 8]
]

print(np.mean(a6_uniform[:, 0]), np.mean(a6_uniform[:, 1]))
print(np.mean(a6_const[:, 0]), np.mean(a6_const[:, 1]))

meanhexes = [np.mean(a, axis=0) for a in hexes]

eff_paths = [
    a0_0kappa[:, 0]*m6_0kappa[:, 0],
    a0_0kappa[:, 1]*m6_0kappa[:, 1],
    a0_0kappa[:, 2]*m6_0kappa[:, 2],
    a0_real[:, 6]*m6_real[:, 6],
    a0_real[:, 7]*m6_real[:, 7],
    a0_real[:, 8]*m6_real[:, 8]
]

eff_meanpaths = [
    np.mean(a0_0kappa[:, 0], axis=0)*np.mean(m6_0kappa[:, 0], axis=0),
    np.mean(a0_0kappa[:, 1], axis=0)*np.mean(m6_0kappa[:, 1], axis=0),
    np.mean(a0_0kappa[:, 2], axis=0)*np.mean(m6_0kappa[:, 2], axis=0),
    np.mean(a0_real[:, 6], axis=0)*np.mean(m6_real[:, 6], axis=0),
    np.mean(a0_real[:, 7], axis=0)*np.mean(m6_real[:, 7], axis=0),
    np.mean(a0_real[:, 8], axis=0)*np.mean(m6_real[:, 8], axis=0)
]

pos = np.array([0, 1, 2, 4, 5, 6])

plt.rcParams.update({'font.size': int(settings.fs - 3)})
plt.bar(pos, meanhexes, color='b')
plt.bar(
    pos,
    eff_meanpaths,
    color=[0.67, 0.85, 0.9]
)

pvals = [
    mannwhitneyu(
        hexes[i], eff_paths[i]
    ).pvalue for i in range(len(hexes))
]
uvals = [
    mannwhitneyu(
        hexes[i], eff_paths[i]
    ).statistic for i in range(len(hexes))
]

for i in range(len(hexes)):
    if i == 1 or i == 4:
        plt.text(
            pos[i],
            1.5*np.max(hexes[i]),
            '(***)',
            horizontalalignment='center'
        )
    if pvals[i] < 0.001:
        plt.text(
            pos[i],
            1.5*np.max(hexes[i]),
            '***',
            horizontalalignment='center'
        )
    elif pvals[i] < 0.01:
        plt.text(
            pos[i],
            1.5*np.max(hexes[i]),
            '**',
            horizontalalignment='center'
        )
    elif pvals[i] < 0.05:
        plt.text(
            pos[i],
            1.5*np.max(hexes[i]),
            '*',
            horizontalalignment='center'
        )
    else:
        plt.text(
            pos[i],
            1.5*np.max(hexes[i]),
            'n.s.',
            horizontalalignment='center'
        )

xoff = 0.2
violins_hexes = plt.violinplot(hexes, pos - xoff, widths=0.35)
violins_paths = plt.violinplot(eff_paths, pos + xoff, widths=0.35)

for pc in violins_paths['bodies']:
    pc.set_facecolor('orange')
    pc.set_alpha(0.5)

# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
violins_paths["cbars"].set_color("orange")
violins_paths["cmins"].set_color("orange")
violins_paths["cmaxes"].set_color("orange")

for pc in violins_hexes['bodies']:
    pc.set_facecolor('red')
    pc.set_alpha(0.5)

violins_hexes["cbars"].set_color("red")
violins_hexes["cmins"].set_color("red")
violins_hexes["cmaxes"].set_color("red")

plt.ylabel('Hexasymmetry (spk/s)')
plt.yscale('log')
plt.xticks(
    pos,
    [
        'star', 'p-l \n standard', 'rand',
        'star', 'p-l \n Gu', 'rand'
    ]
)
plt.ylim(0.1, 100)
plt.tight_layout()

plot_fname = os.path.join(
    settings.loc,
    "clustering",
    "no-clustering",
    "figure_noclustering.png"
)
os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(plot_fname, dpi=300)
