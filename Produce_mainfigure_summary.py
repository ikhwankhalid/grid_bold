import numpy as np
import matplotlib.pyplot as plt
import settings
import os
from scipy.stats import mannwhitneyu


###############################################################################
# Functions                                                                   #
###############################################################################
def compute_total_from_files(files_path):
    """
    Determines total simulations based on existing file names.
    """
    total = 0
    files_path = os.path.join(settings.loc, "summary", "ideal")
    # Counts only files that match the expected naming convention
    for fname in os.listdir(files_path):
        num = int(fname.split(".")[0].split("_")[-1].split("-")[-1])
        if num > total:
            total = num

    return total


###############################################################################
# Plotting                                                                    #
###############################################################################
# Dynamically compute 'total' before initializing arrays
total = compute_total_from_files(
    os.path.join(settings.loc, "summary", "ideal")
)

# Load data from files
a6_ideal = np.zeros((total, 10))
m6_ideal = np.zeros((total, 10))
a0_ideal = np.zeros((total, 10))
a6_real = np.zeros((total, 10))
m6_real = np.zeros((total, 10))
a0_real = np.zeros((total, 10))
for i in np.arange(0, total, settings.rep):
    a = np.load(
            os.path.join(
                settings.loc,
                "summary",
                "ideal",
                "summary_bar_plot_3hyp3traj_" +
                str(i+1) + '-' + str(i + settings.rep) + '.npy'
            ),
            allow_pickle=True
        )
    b = np.load(
            os.path.join(
                settings.loc,
                "summary",
                "real",
                "summary_bar_plot_3hyp3traj_" +
                str(i+1) + '-' + str(i + settings.rep) + '.npy'
            ),
            allow_pickle=True
        )
    a = np.reshape(a, (10, 3, settings.rep))
    b = np.reshape(b, (10, 3, settings.rep))

    a6_ideal[i:(i+settings.rep), :] = a[:, 0, :].T
    m6_ideal[i:(i+settings.rep), :] = a[:, 1, :].T
    a0_ideal[i:(i+settings.rep), :] = a[:, 2, :].T
    a6_real[i:(i+settings.rep), :] = b[:, 0, :].T
    m6_real[i:(i+settings.rep), :] = b[:, 1, :].T
    a0_real[i:(i+settings.rep), :] = b[:, 2, :].T


a6_ideal[:, 3:9] = np.roll(a6_ideal[:, 3:9], 3, 1)
m6_ideal[:, 3:9] = np.roll(m6_ideal[:, 3:9], 3, 1)
a0_ideal[:, 3:9] = np.roll(a0_ideal[:, 3:9], 3, 1)
a6_real[:, 3:9] = np.roll(a6_real[:, 3:9], 3, 1)
m6_real[:, 3:9] = np.roll(m6_real[:, 3:9], 3, 1)
a0_real[:, 3:9] = np.roll(a0_real[:, 3:9], 3, 1)

xoff = 0.2
pos = np.array([
    0, 1, 2, 3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11, 12, 13, 15, 16, 17, 18.5, 19.5,
    20.5
])

ideal_pos = pos[[0, 1, 2, 6, 7, 8, 12, 13, 14]]
real_pos = pos[[3, 4, 5, 9, 10, 11, 15, 16, 17]]
fig = plt.figure(figsize=(18, 8))
plt.rcParams.update({'font.size': int(settings.fs - 3)})

randoffs = np.random.normal(0., 0.05, total)
plt.bar(ideal_pos, np.mean(a6_ideal[:, :9], axis=0), color='b')
plt.bar(real_pos, np.mean(a6_real[:, :9], axis=0), color='b')
plt.xticks(pos,
           ['star', 'p-l \n ideal', 'rand',
            'star', 'p-l \n Doeller', 'rand',
            'star', 'p-l \n ideal', 'rand',
            'star', 'p-l \n weaker', 'rand',
            'star', 'p-l \n ideal', 'rand',
            'star', 'p-l \n Gu', 'rand'])
plt.bar(
    ideal_pos,
    np.mean(m6_ideal[:, :9], axis=0) * np.mean(a0_ideal[:, :9], axis=0),
    color=[0.67, 0.85, 0.9]
)
plt.bar(
    real_pos,
    np.mean(m6_real[:, :9], axis=0) * np.mean(a0_real[:, :9], axis=0),
    color=[0.67, 0.85, 0.9]
)  # , edgecolor='k')
plt.ylabel('Hexasymmetry (spk/s)')
plt.yscale('log')

hexes = [
    a6_ideal[:, 0],
    a6_ideal[:, 1],
    a6_ideal[:, 2],
    a6_real[:, 0],
    a6_real[:, 1],
    a6_real[:, 2],
    a6_ideal[:, 3],
    a6_ideal[:, 4],
    a6_ideal[:, 5],
    a6_real[:, 3],
    a6_real[:, 4],
    a6_real[:, 5],
    a6_ideal[:, 6],
    a6_ideal[:, 7],
    a6_ideal[:, 8],
    a6_real[:, 6],
    a6_real[:, 7],
    a6_real[:, 8]
]

for i in range(len(hexes)):
    print(f"{np.mean(hexes[i]):.2f}")

violins_hexes = plt.violinplot(hexes, pos - xoff, widths=0.35)

eff_paths = [
    a0_ideal[:, 0]*m6_ideal[:, 0],
    a0_ideal[:, 1]*m6_ideal[:, 1],
    a0_ideal[:, 2]*m6_ideal[:, 2],
    a0_real[:, 0]*m6_real[:, 0],
    a0_real[:, 1]*m6_real[:, 1],
    a0_real[:, 2]*m6_real[:, 2],
    a0_ideal[:, 3]*m6_ideal[:, 3],
    a0_ideal[:, 4]*m6_ideal[:, 4],
    a0_ideal[:, 5]*m6_ideal[:, 5],
    a0_real[:, 3]*m6_real[:, 3],
    a0_real[:, 4]*m6_real[:, 4],
    a0_real[:, 5]*m6_real[:, 5],
    a0_ideal[:, 6]*m6_ideal[:, 6],
    a0_ideal[:, 7]*m6_ideal[:, 7],
    a0_ideal[:, 8]*m6_ideal[:, 8],
    a0_real[:, 6]*m6_real[:, 6],
    a0_real[:, 7]*m6_real[:, 7],
    a0_real[:, 8]*m6_real[:, 8],
]

violins_paths = plt.violinplot(eff_paths, pos + xoff, widths=0.35)

for pc in violins_paths['bodies']:
    pc.set_facecolor('orange')
    # pc.set_edgecolor('orange')
    pc.set_alpha(0.5)

# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
violins_paths["cbars"].set_color("orange")
violins_paths["cmins"].set_color("orange")
violins_paths["cmaxes"].set_color("orange")

for pc in violins_hexes['bodies']:
    pc.set_facecolor('red')
    # pc.set_edgecolor('red')
    pc.set_alpha(0.5)

# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
violins_hexes["cbars"].set_color("red")
violins_hexes["cmins"].set_color("red")
violins_hexes["cmaxes"].set_color("red")


# add asterisks for statistical significance
alldat_fr = np.c_[
    a6_ideal[:, :3],
    a6_real[:, :3],
    a6_ideal[:, 3:6],
    a6_real[:, 3:6],
    a6_ideal[:, 6:9],
    a6_real[:, 6:9]
]

alldat_path = np.c_[
    m6_ideal[:, :3],
    m6_real[:, :3],
    m6_ideal[:, 3:6],
    m6_real[:, 3:6],
    m6_ideal[:, 6:9],
    m6_real[:, 6:9]
] * np.c_[
    a0_ideal[:, :3],
    a0_real[:, :3],
    a0_ideal[:, 3:6],
    a0_real[:, 3:6],
    a0_ideal[:, 6:9],
    a0_real[:, 6:9]
]
pvals = [
    mannwhitneyu(
        alldat_fr[:, i], alldat_path[:, i]
    ).pvalue for i in range(18)
]
uvals = [
    mannwhitneyu(
        alldat_fr[:, i], alldat_path[:, i]
    ).statistic for i in range(18)
]

for i in range(18):
    if i == 13 or i == 14 or i == 16:
        plt.text(
            pos[i],
            1.5*np.max(alldat_fr[:, i]),
            '(***)',
            horizontalalignment='center'
        )
    if pvals[i] < 0.001:
        plt.text(
            pos[i],
            1.5*np.max(alldat_fr[:, i]),
            '***',
            horizontalalignment='center'
        )
    elif pvals[i] < 0.01:
        plt.text(
            pos[i],
            1.5*np.max(alldat_fr[:, i]),
            '**',
            horizontalalignment='center'
        )
    elif pvals[i] < 0.05:
        plt.text(
            pos[i],
            1.5*np.max(alldat_fr[:, i]),
            '*',
            horizontalalignment='center'
        )
    else:
        plt.text(
            pos[i],
            1.5*np.max(alldat_fr[:, i]),
            'n.s.',
            horizontalalignment='center'
        )


pval_repsupp_starpwl = mannwhitneyu(
    alldat_fr[:, 12], alldat_fr[:, 13]
).pvalue
uval_repsupp_starpwl = mannwhitneyu(
    alldat_fr[:, 12], alldat_fr[:, 13]
).statistic

pval_clust_starpwl = mannwhitneyu(alldat_fr[:, 6], alldat_fr[:, 7]).pvalue
uval_clust_starpwl = mannwhitneyu(alldat_fr[:, 6], alldat_fr[:, 7]).statistic

plt.ylim([1e-1, 1e4])
plt.axvline(x=(pos[5] + pos[6])/2, color='k')
plt.axvline(x=(pos[11] + pos[12])/2, color='k')
plt.text(
    (pos[2] + pos[3])/2,
    3e3,
    'Conjunctive',
    fontsize=settings.fs - 3,
    horizontalalignment='center'
)
plt.text(
    (pos[8] + pos[9])/2,
    3e3,
    'Repetition suppression',
    fontsize=settings.fs - 3,
    horizontalalignment='center'
)
plt.text(
    (pos[12] + pos[17])/2,
    3e3,
    'Structure-function mapping',
    fontsize=settings.fs - 3,
    horizontalalignment='center'
)

plt.savefig(
    'summary_figure.png', dpi=300, bbox_inches='tight'
)
plt.close(fig)
