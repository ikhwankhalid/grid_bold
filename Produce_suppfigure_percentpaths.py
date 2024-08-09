import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import matplotlib.colors as mcol
import settings
import os
from utils.utils import colorline
from utils.grid_funcs import traj
from numba import jit
from tqdm import tqdm
from Produce_mainfigure_summary import compute_total_from_files

###############################################################################
# Parameters                                                                  #
###############################################################################
pwl_angs = np.linspace(- np.pi, np.pi, settings.phbins, endpoint=False)
t6_fname = os.path.join(
    settings.loc,
    "summary",
    "clust_t6.pkl"
)
os.makedirs(
    os.path.dirname(t6_fname),
    exist_ok=True
)

###############################################################################
# Run                                                                         #
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

if not os.path.exists(t6_fname):
    @jit
    def get_t6bars(n_trials, prop_list, pwl_angs):
        pwl_t6 = np.zeros((n_trials, len(prop_list)))
        rw_t6 = np.zeros((n_trials, len(prop_list)))
        for id1 in tqdm(range(n_trials)):
            _, _, _, rw_direcs = traj(settings.dt, settings.tmax)
            for id2, p in enumerate(prop_list):
                n_pwlangs = int(settings.phbins * p / 100)
                n_rwdirs = int(len(rw_direcs) * p / 100)

                pwl_subangs = np.random.choice(
                    pwl_angs, n_pwlangs, replace=False
                )
                rw_subdirecs = np.random.choice(
                    rw_direcs, n_rwdirs, replace=False
                )

                pwl_t6[id1, id2] = np.abs(
                    np.sum(np.exp(- 6 * pwl_subangs * 1j))
                ) / (len(pwl_subangs))
                rw_t6[id1, id2] = np.abs(
                    np.sum(np.exp(- 6 * rw_subdirecs * 1j))
                ) / (len(rw_subdirecs))

        return pwl_t6, rw_t6

    pwl_t6, rw_t6 = get_t6bars(360, settings.prop_list, pwl_angs)
    with open(t6_fname, "wb") as f:
        pickle.dump((pwl_t6, rw_t6), f)
else:
    with open(t6_fname, "rb") as f:
        pwl_t6, rw_t6 = pickle.load(f)


fig = plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': int(settings.fs - 3)})

zs = (settings.phbins * settings.prop_list / 100)

barwidth = 0.25
pos = np.linspace(1, 3, 4, endpoint=True)
xs = np.linspace(pos[0] - 0.2, pos[0] - 0.2 + 1e-6, pwl_t6.shape[-1])
xs2 = np.linspace(pos[1] - 0.2, pos[1] - 0.2 + 1e-6, pwl_t6.shape[-1])
xs3 = np.linspace(pos[2] - 0.2, pos[2] - 0.2 + 1e-6, pwl_t6.shape[-1])
xs4 = np.linspace(pos[3] - 0.2, pos[3] - 0.2 + 1e-6, pwl_t6.shape[-1])
ys = np.mean(pwl_t6, axis=0) * np.mean(a0_ideal[:, 7])
ys2 = np.mean(rw_t6, axis=0) * np.mean(a0_ideal[:, 8])
ys3 = np.mean(pwl_t6, axis=0) * np.mean(a0_real[:, 7])
ys4 = np.mean(rw_t6, axis=0) * np.mean(a0_real[:, 8])


plt.bar(pos[0], np.mean(a6_ideal[:, 7], axis=0), width=barwidth, color='b')
plt.bar(
    pos[0],
    np.mean(m6_ideal[:, 7], axis=0) * np.mean(a0_ideal[:, 7], axis=0),
    width=barwidth,
    color=[0.67, 0.85, 0.9]
)

plt.bar(pos[1], np.mean(a6_ideal[:, 8], axis=0), width=barwidth, color='b')
plt.bar(
    pos[1],
    np.mean(m6_ideal[:, 8], axis=0) * np.mean(a0_ideal[:, 8], axis=0),
    width=barwidth,
    color=[0.67, 0.85, 0.9]
)


plt.bar(pos[2], np.mean(a6_real[:, 7], axis=0), width=barwidth, color='b')
plt.bar(
    pos[2],
    np.mean(m6_real[:, 7], axis=0) * np.mean(a0_real[:, 7], axis=0),
    width=barwidth,
    color=[0.67, 0.85, 0.9]
)


plt.bar(pos[3], np.mean(a6_real[:, 8], axis=0), width=barwidth, color='b')
plt.bar(
    pos[3],
    np.mean(m6_real[:, 8], axis=0) * np.mean(a0_real[:, 8], axis=0),
    width=barwidth,
    color=[0.67, 0.85, 0.9]
)

print(np.amin(ys2), np.amin(ys4))
print(
    np.mean(m6_ideal[:, 8], axis=0) * np.mean(a0_ideal[:, 8], axis=0),
    np.mean(m6_real[:, 8], axis=0) * np.mean(a0_real[:, 8], axis=0)
)
colorline(
    xs,
    ys,
    settings.prop_list,
    cmap=plt.get_cmap('turbo'),
    linewidth=15,
    norm=mcol.LogNorm(1, 100)
)
colorline(
    xs2,
    ys2,
    settings.prop_list,
    cmap=plt.get_cmap('turbo'),
    linewidth=15,
    norm=mcol.LogNorm(1, 100)
)
colorline(
    xs3,
    ys3,
    settings.prop_list,
    cmap=plt.get_cmap('turbo'),
    linewidth=15,
    norm=mcol.LogNorm(1, 100)
)
colorline(
    xs4,
    ys4,
    settings.prop_list,
    cmap=plt.get_cmap('turbo'),
    linewidth=15,
    norm=mcol.LogNorm(1, 100)
)

plt.yscale("log")
plt.ylabel('Hexasymmetry (spk/s)')
plt.ylim(1e-1, 1e3)
plt.xlim(0.5, 3.3)
plt.xticks([])
plt.xticks(
    pos,
    ['p-l', 'rand', 'p-l', 'rand']
)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.gcf().text(0.275, 0, "ideal", fontsize=settings.fs)
plt.gcf().text(0.625, 0, "Gu", fontsize=settings.fs)

cmap = mpl.cm.turbo
norm = mpl.colors.Normalize(vmin=1, vmax=100)
ax = plt.gca()
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    label='Percentage of subsampled path segments',
    ax=ax
)
plt.yticks([1, 50, 100])
plt.tight_layout()

plot_fname = os.path.join(
    settings.loc,
    "clustering",
    "percentpaths",
    "Figure_percentpaths.png"
)
os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(plot_fname, dpi=300)
