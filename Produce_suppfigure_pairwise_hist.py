import numpy as np
import matplotlib.pyplot as plt
import settings
import os
from scipy.stats import mannwhitneyu


# Example function to determine total based on file count
def compute_total_from_files(files_path):
    total = 0
    files_path = os.path.join(settings.loc, "summary", "ideal")
    # Counts only files that match the expected naming convention
    for fname in os.listdir(files_path):
        num = int(fname.split(".")[0].split("_")[-1].split("-")[-1])
        if num > total:
            total = num

    return total


# Dynamically compute 'total' before initializing arrays
total = compute_total_from_files(
    os.path.join(settings.loc, "summary", "ideal")
)

# load data from files
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


alldat_fr = np.c_[
    a6_ideal[:, :3],
    a6_real[:, :3],
    a6_ideal[:, 3:6],
    a6_real[:, 3:6],
    a6_ideal[:, 6:9],
    a6_real[:, 6:9]
]


fig = plt.figure(figsize=(14, 16))
plt.rcParams.update({'font.size': settings.fs})
spec = fig.add_gridspec(
    ncols=3,
    nrows=6,
)

# plot hexsyms
for i in range(6):
    dataslice = alldat_fr[:, i * 3:(i * 3) + 3]
    choice = [(0, 1), (0, 2), (1, 2)]
    colour = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["star", "p-l", "rand"]

    for j in range(3):
        fig.add_subplot(spec[i, j])
        bins = np.histogram(
            np.hstack(
                (dataslice[:, choice[j][0]], dataslice[:, choice[j][1]])
            ), bins=30
        )[1]
        plt.hist(
            dataslice[:, choice[j][0]],
            bins=bins,
            alpha=0.6,
            histtype='bar',
            ec='black',
            color=colour[choice[j][0]],
            label=labels[choice[j][0]]
        )
        plt.hist(
            dataslice[:, choice[j][1]],
            bins=bins,
            alpha=0.6,
            histtype='bar',
            ec='black',
            color=colour[choice[j][1]],
            label=labels[choice[j][0]]
        )
        if i == 2 and j == 0:
            plt.hist(
                a6_ideal[:, 9],
                bins=bins,
                histtype='bar',
                ls="dashed",
                lw=1.5,
                ec="#1f77b4",
                zorder=0,
                fc=(0, 0, 1, 0)
            )
        if i == 3 and j == 0:
            plt.hist(
                a6_real[:, 9],
                bins=bins,
                histtype='bar',
                ls="dashed",
                lw=1.5,
                ec="#1f77b4",
                zorder=0,
                fc=(0, 0, 1, 0)
            )
        if i == 2 and j == 1:
            plt.hist(
                a6_ideal[:, 9],
                bins=bins,
                histtype='bar',
                ls="dashed",
                lw=1.5,
                ec="#1f77b4",
                zorder=0,
                fc=(0, 0, 1, 0)
            )
        if i == 3 and j == 1:
            plt.hist(
                a6_real[:, 9],
                bins=bins,
                histtype='bar',
                ls="dashed",
                lw=1.5,
                ec="#1f77b4",
                zorder=0,
                fc=(0, 0, 1, 0)
            )
        plt.yticks([])

        pval = mannwhitneyu(
            dataslice[:, choice[j][0]], dataslice[:, choice[j][1]]
        ).pvalue
        uval = mannwhitneyu(
            dataslice[:, choice[j][0]], dataslice[:, choice[j][1]]
        ).statistic

        if pval < 0.001:
            fig.axes[i*3 + j].text(
                0.7,
                0.85,
                '***',
                ma='center',
                transform=fig.axes[i*3 + j].transAxes
            )
        elif pval < 0.01:
            fig.axes[i*3 + j].text(
                0.7,
                0.85,
                '**',
                ma='center',
                transform=fig.axes[i*3 + j].transAxes
            )
        elif pval < 0.05:
            fig.axes[i*3 + j].text(
                0.7,
                0.85,
                '*',
                ma='center',
                transform=fig.axes[i*3 + j].transAxes
            )
        else:
            fig.axes[i*3 + j].text(
                0.7,
                0.85,
                'n.s.',
                ma='center',
                transform=fig.axes[i*3 + j].transAxes
            )

        fig.axes[i*3 + j].spines['top'].set_visible(False)
        fig.axes[i*3 + j].spines['right'].set_visible(False)

xoffs = 0.075
labels = [(0.055 - xoffs, 0.73, "Conjunctive"),
          (0.098 - xoffs, 0.81, "Ideal"),
          (0.098 - xoffs, 0.66, "Doeller"),
          (0.055 - xoffs, 0.42, "Repetition-suppression"),
          (0.098 - xoffs, 0.55, "Ideal"),
          (0.098 - xoffs, 0.41, "Weaker"),
          (0.055 - xoffs, 0.15, "Structure-function\n       mapping"),
          (0.098 - xoffs, 0.28, "Ideal"),
          (0.098 - xoffs, 0.16, "Gu"),
          (0.09, 0.46, "Probability")]

for x, y, label in labels:
    plt.text(
        x,
        y,
        label,
        fontsize=settings.fs,
        transform=plt.gcf().transFigure,
        rotation=90
    )

fig.axes[-1].set_xlabel("p-l/rand", labelpad=65)
fig.axes[-2].set_xlabel("star/rand", labelpad=65)
fig.axes[-3].set_xlabel("star/p-l", labelpad=65)

plt.text(
    0.41,
    0.06,
    "Hexasymmetry (spk/s)",
    fontsize=settings.fs,
    transform=plt.gcf().transFigure
)

lines1, labels1 = fig.axes[0].get_legend_handles_labels()
lines2, labels2 = fig.axes[1].get_legend_handles_labels()
lines2.reverse()
lines1 = lines1 + lines2
# fig.legend(lines, labels, loc='upper center')

labels = ["star", "p-l", "rand"]
fig.legend(
    lines1[:3],
    labels,
    loc='center right',
    bbox_to_anchor=(1.05, 0.5),
    ncol=1,
    bbox_transform=fig.transFigure
)

plt.tight_layout()
plt.savefig("pairwise_hist.png", dpi=300, bbox_inches="tight")
plt.close()
