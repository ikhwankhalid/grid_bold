import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcol
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.utils import colorline, get_pathsym
from functions.gridfcts import traj, traj_pwl
import utils.settings as settings
import os
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from numba import jit, njit
import pickle
import itertools
from tqdm import tqdm


def heatmap(data, row_labels, col_labels, xlabel='', ylabel='', extralines=True, ax1=None,
            cbar_kw={}, cbarnorm=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax1:
        ax1 = plt.gca()

    # Plot the heatmap
    im1 = ax1.imshow(data, norm=cbarnorm, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax1.figure.colorbar(im1, cax=cax, **cbar_kw)
    #cbar.ax.set_yticklabels(['repulsion', 'attraction'])
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=10)

    # We want to show all ticks...
    ax1.set_xticks(np.arange(data.shape[1]))
    ax1.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax1.set_xticklabels(col_labels)
    ax1.set_yticklabels(row_labels)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    # Let the horizontal axes labeling appear on top.
    ax1.tick_params(top=False, bottom=True,
                    labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=-45, ha="left", va='top',
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax1.spines.items():
        spine.set_visible(False)

    ax1.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax1.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax1.grid(which="minor", color="w", linestyle='-', linewidth=2)
    if extralines:
        ax1.axvline(10.5, linestyle='--', color='k')
        ax1.axvline(21.5, linestyle='--', color='k')
        ax1.axvline(32.5, linestyle='--', color='k')
        ax1.axvline(43.5, linestyle='--', color='k')
        ax1.axvline(54.5, linestyle='--', color='k')
        ax1.axvline(65.5, linestyle='--', color='k')

    ax1.tick_params(which="minor", bottom=False, left=False)

    return im1, cbar


def annotate_heatmap(im1, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, white=False, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im1.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im1.norm(threshold)
    else:
        threshold = im1.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw1 = dict(horizontalalignment="center",
               verticalalignment="center")
    kw1.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #kw.update(color=textcolors[int(im1.norm(data[i, j]) > threshold)])
            if data[i, j] < 1e-3 and white:
                kw1.update(color='white')
            else:
                kw1.update(color='black')
            text = im1.axes.text(j, i, valfmt(data[i, j], None), **kw1)
            texts.append(text)
    return texts

def func(xval, pos):
    """
    helper function for the annotation of correlation values
    """
    return "{:.1f}".format(xval).replace("0.", ".").replace("1.0", "1.")

def func2(xval, pos):
    """
    helper function for the annotation of p-values
    """
    xval2 = -np.log10(xval)
    #return "" if (xval2 > 99 or xval == 1) else "{:.0f}".format(xval2)
    return "" if (xval2 > 99) else "{:.0f}".format(xval2)


###############################################################################
# Summary figure                                                              #
###############################################################################
total = 360


# load data from files
a6_ideal = np.zeros((total, 10))
m6_ideal = np.zeros((total, 10))
a0_ideal = np.zeros((total, 10))
a6_real = np.zeros((total, 10))
m6_real = np.zeros((total, 10))
a0_real = np.zeros((total, 10))
for i in np.arange(0, total, settings.rep):
    a = np.load(os.path.join(settings.loc, "summary", "summary_bar_plot_3hyp3traj_ideal_" + str(i+1) + '-' + str(i+settings.rep) + '.npy'), allow_pickle=True)
    b = np.load(os.path.join(settings.loc, "summary", "summary_bar_plot_3hyp3traj_real_" + str(i+1) + '-' + str(i+settings.rep) + '.npy'), allow_pickle=True)
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


# print(a0_ideal)
# print(a0_real)


# rep = total
xoff = 0.2
pos = np.array([0, 1, 2, 3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11, 12, 13, 15, 16, 17, 18.5, 19.5, 20.5])
# pos[13:] += 0.75
# pos[14:] += 0.75
# pos[3:] += 0.5
# pos[9:] += 0.5
# pos[15:] += 0.5
ideal_pos = pos[[0, 1, 2, 6, 7, 8, 12, 13, 14]]
real_pos = pos[[3, 4, 5, 9, 10, 11, 15, 16, 17]]
fig = plt.figure(figsize=(18, 8))
plt.rcParams.update({'font.size': int(settings.fs - 3)})

randoffs = np.random.normal(0., 0.05, total)
plt.bar(ideal_pos, np.mean(a6_ideal[:, :9], axis=0), color='b')
plt.bar(real_pos, np.mean(a6_real[:, :9], axis=0), color='b')
# plt.plot(pos[0] * np.zeros(total)-xoff + randoffs, a6_ideal[:, 0], 'k.')
# plt.plot(pos[1] * np.ones(total)-xoff + randoffs, a6_ideal[:, 1], 'k.')
# plt.plot(pos[2] *np.ones(total)-xoff + randoffs, a6_ideal[:, 2], 'k.')
# plt.plot(pos[3] *np.ones(total)-xoff + randoffs, a6_real[:, 0], 'k.')
# plt.plot(pos[4] *np.ones(total)-xoff + randoffs, a6_real[:, 1], 'k.')
# plt.plot(pos[5] *np.ones(total)-xoff + randoffs, a6_real[:, 2], 'k.')
# plt.plot(pos[6] *np.ones(total)-xoff + randoffs, a6_ideal[:, 3], 'k.')
# plt.plot(pos[7] *np.ones(total)-xoff + randoffs, a6_ideal[:, 4], 'k.')
# plt.plot(pos[8] *np.ones(total)-xoff + randoffs, a6_ideal[:, 5], 'k.')
# plt.plot(pos[9] *np.ones(total)-xoff + randoffs, a6_real[:, 3], 'k.')
# plt.plot(pos[10] *np.ones(total)-xoff + randoffs, a6_real[:, 4], 'k.')
# plt.plot(pos[11] *np.ones(total)-xoff + randoffs, a6_real[:, 5], 'k.')
# plt.plot(pos[12] *np.ones(total)-xoff + randoffs, a6_ideal[:, 6], 'k.')
# plt.plot(pos[13] *np.ones(total)-xoff + randoffs, a6_ideal[:, 7], 'k.')
# plt.plot(pos[14] *np.ones(total)-xoff + randoffs, a6_ideal[:, 8], 'k.')
# plt.plot(pos[15] *np.ones(total)-xoff + randoffs, a6_real[:, 6], 'k.')
# plt.plot(pos[16] *np.ones(total)-xoff + randoffs, a6_real[:, 7], 'k.')
# plt.plot(pos[17] *np.ones(total)-xoff + randoffs, a6_real[:, 8], 'k.')

# plt.scatter(pos[0] * np.zeros(total)+xoff + randoffs, a0_ideal[:, 0]*m6_ideal[:, 0], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[1] * np.ones(total)+xoff + randoffs, a0_ideal[:, 1]*m6_ideal[:, 1], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[2] * np.ones(total)+xoff + randoffs, a0_ideal[:, 2]*m6_ideal[:, 2], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[3] * np.ones(total)+xoff + randoffs, a0_real[:, 0]*m6_real[:, 0], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[4] * np.ones(total)+xoff + randoffs, a0_real[:, 1]*m6_real[:, 1], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[5] * np.ones(total)+xoff + randoffs, a0_real[:, 2]*m6_real[:, 2], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[6] * np.ones(total)+xoff + randoffs, a0_ideal[:, 3]*m6_ideal[:, 3], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[7] * np.ones(total)+xoff + randoffs, a0_ideal[:, 4]*m6_ideal[:, 4], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[8] * np.ones(total)+xoff + randoffs, a0_ideal[:, 5]*m6_ideal[:, 5], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[9] * np.ones(total)+xoff + randoffs, a0_real[:, 3]*m6_real[:, 3], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[10] * np.ones(total)+xoff + randoffs, a0_real[:, 4]*m6_real[:, 4], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[11] * np.ones(total)+xoff + randoffs, a0_real[:, 5]*m6_real[:, 5], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[12] * np.ones(total)+xoff + randoffs, a0_ideal[:, 6]*m6_ideal[:, 6], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[13] * np.ones(total)+xoff + randoffs, a0_ideal[:, 7]*m6_ideal[:, 7], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[14] * np.ones(total)+xoff + randoffs, a0_ideal[:, 8]*m6_ideal[:, 8], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[15] * np.ones(total)+xoff + randoffs, a0_real[:, 6]*m6_real[:, 6], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[16] * np.ones(total)+xoff + randoffs, a0_real[:, 7]*m6_real[:, 7], edgecolor='k', facecolor='none', zorder=5)
# plt.scatter(pos[17] * np.ones(total)+xoff + randoffs, a0_real[:, 8]*m6_real[:, 8], edgecolor='k', facecolor='none', zorder=5)
plt.xticks(pos,
           ['star', 'p-l \n ideal', 'rand',
            'star', 'p-l \n Doeller', 'rand',
            'star', 'p-l \n ideal', 'rand',
            'star', 'p-l \n weaker', 'rand',
            'star', 'p-l \n ideal', 'rand',   
            'star', 'p-l \n Gu', 'rand']) 
plt.bar(ideal_pos, np.mean(m6_ideal[:, :9], axis=0) * np.mean(a0_ideal[:, :9], axis=0), color=[0.67, 0.85, 0.9]) #, edgecolor='k')
plt.bar(real_pos, np.mean(m6_real[:, :9], axis=0) * np.mean(a0_real[:, :9], axis=0), color=[0.67, 0.85, 0.9])#, edgecolor='k')
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


# Make all the violin statistics marks red:
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
#     vp1 = violins_hexes[partname]
#     vp1.set_edgecolor("black")
#     vp1.set_linewidth(1)
#     vp2 = violins_paths[partname]
#     vp2.set_edgecolor("black")
#     vp2.set_linewidth(1)

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
alldat_fr = np.c_[a6_ideal[:, :3], a6_real[:, :3], a6_ideal[:, 3:6], a6_real[:, 3:6], a6_ideal[:, 6:9], a6_real[:, 6:9]]
alldat_path = np.c_[m6_ideal[:, :3], m6_real[:, :3], m6_ideal[:, 3:6], m6_real[:, 3:6], m6_ideal[:, 6:9], m6_real[:, 6:9]] * np.c_[a0_ideal[:, :3], a0_real[:, :3], a0_ideal[:, 3:6], a0_real[:, 3:6], a0_ideal[:, 6:9], a0_real[:, 6:9]]
pvals = [mannwhitneyu(alldat_fr[:, i], alldat_path[:, i]).pvalue for i in range(18)]
uvals = [mannwhitneyu(alldat_fr[:, i], alldat_path[:, i]).statistic for i in range(18)]
# pvals = [mannwhitneyu(alldat_fr[~np.isnan(alldat_fr).any(axis=0)][:, i], alldat_path[~np.isnan(alldat_path).any(axis=0)][:, i]).pvalue for i in range(18)]
# uvals = [mannwhitneyu(alldat_fr[~np.isnan(alldat_fr).any(axis=0)][:, i], alldat_path[~np.isnan(alldat_path).any(axis=0)][:, i]).statistic for i in range(18)]

for i in range(18):
    if i == 13 or i == 14 or i == 16:
        plt.text(pos[i], 1.5*np.max(alldat_fr[:, i]), '(***)', horizontalalignment='center')
    if pvals[i] < 0.001:
        plt.text(pos[i], 1.5*np.max(alldat_fr[:, i]), '***', horizontalalignment='center')
    elif pvals[i] < 0.01:
        plt.text(pos[i], 1.5*np.max(alldat_fr[:, i]), '**', horizontalalignment='center')
    elif pvals[i] < 0.05:
        plt.text(pos[i], 1.5*np.max(alldat_fr[:, i]), '*', horizontalalignment='center')
    else:
        plt.text(pos[i], 1.5*np.max(alldat_fr[:, i]), 'n.s.', horizontalalignment='center')


pval_repsupp_starpwl = mannwhitneyu(alldat_fr[:, 12], alldat_fr[:, 13]).pvalue
uval_repsupp_starpwl = mannwhitneyu(alldat_fr[:, 12], alldat_fr[:, 13]).statistic

pval_clust_starpwl = mannwhitneyu(alldat_fr[:, 6], alldat_fr[:, 7]).pvalue
uval_clust_starpwl = mannwhitneyu(alldat_fr[:, 6], alldat_fr[:, 7]).statistic


# pairwise comparison clustering
# # plt.plot([15, 16], 10**2*np.ones(2), 'k-')
# plt.plot([15, 15+1e-5, 16, 16+1e-5], [10**2 - 2*10**1, 10**2, 10**2, 10**2 - 2*10**1], 'k-')
# plt.plot([15.5, 15.5+1e-5], [10**2, 10**2 + 2*10**1], 'k-')
# plt.text(15.5, 10**2 + 3*10**1, '***', horizontalalignment='center')

# # plt.plot([7.5, 8.5], (10**2 + 1*10**2)*np.ones(2), 'k-')
# plt.plot([7.5, 7.5+1e-5, 8.5, 8.5+1e-5], [(10**2 + 1*10**2) - 4*10**1, (10**2 + 1*10**2), (10**2 + 1*10**2), (10**2 + 1*10**2) - 4*10**1], 'k-')
# plt.plot([8, 8+1e-5], [(10**2 + 1*10**2), (10**2 + 1*10**2) + 4*10**1], 'k-')
# plt.text(8, (10**2 + 1*10**2) + 6*10**1, '***', horizontalalignment='center')

plt.ylim([1e-1, 1e4])
plt.axvline(x=(pos[5] + pos[6])/2, color='k')
plt.axvline(x=(pos[11] + pos[12])/2, color='k')
plt.text((pos[2] + pos[3])/2, 3e3, 'Conjunctive', fontsize=settings.fs - 3, horizontalalignment='center')
plt.text((pos[8] + pos[9])/2, 3e3, 'Repetition suppression', fontsize=settings.fs - 3, horizontalalignment='center')
plt.text((pos[12] + pos[17])/2, 3e3, 'Structure-function mapping', fontsize=settings.fs - 3, horizontalalignment='center')


pwl_angs = np.linspace(- np.pi, np.pi, settings.phbins, endpoint=False)


t6_fname = os.path.join(
    settings.loc,
    "summary",
    f"clust_t6.pkl"
)
os.makedirs(
    os.path.dirname(t6_fname),
    exist_ok=True
)
plot_fname = os.path.join(settings.loc, "plots", 'summary_figure_all.png')
os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(
    plot_fname,
    dpi=300
)
plt.close(fig)

data = np.c_[a6_ideal / a0_ideal / m6_ideal, a6_real / a0_real / m6_real]
pvals_all = np.ones((18, 18))
for i in range(18):
    for j in range(18):
            if i == j:
                continue
            pvals_all[i, j] = mannwhitneyu(data[:, i], data[:, j]).pvalue

LABELS1 = ['star conj ideal', 'pl conj ideal', 'rw conj ideal',
           'star conj D', 'pl conj D', 'rw conj D',
           'star clust ideal', 'pl clust ideal', 'rw clust ideal',
           'star clust G', 'pl clust G', 'rw clust G',
           'star rep ideal', 'pl rep ideal', 'rw rep ideal',
           'star rep weak', 'pl rep weak', 'rw rep weak']

pvals = np.tril(pvals)

plt.figure(figsize=(12, 12))
AX1 = plt.subplot(1, 1, 1)
REDSREV = cm.get_cmap('Reds_r')
IM, _ = heatmap(
    pvals, LABELS1, LABELS1, extralines=False, ax1=AX1,
    cmap=REDSREV, vmin=1e-6, vmax=1,
    # cbarnorm=mpl.colors.LogNorm(),
    cbarlabel="p value"
)
plt.close()


plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': int(settings.fs - 3)})
plt.hist(alldat_fr[:, 12], bins=30, range=(16, 19), alpha=0.6, label="Star-like walk", histtype='bar', ec='black')
plt.hist(alldat_fr[:, 13], bins=30, range=(16, 19), alpha=0.6, label="Piece-wise linear walk", histtype='bar', ec='black')
plt.title("Repetition-suppression")
plt.xlabel("Hexasymmetry (Spk/s)")
plt.ylabel("Count")
plt.text(18., 50, f"p: {np.round(pval_repsupp_starpwl, 7)}\nu: {uval_repsupp_starpwl}")
plot_fname = os.path.join(settings.loc, "plots", "hist_repsupp_starpwl.png")
plt.savefig(
    plot_fname,
    dpi=300
)
plt.close()


###############################################################################
# Pairwise comparisons                                                        #
###############################################################################
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
        bins=np.histogram(np.hstack((dataslice[:, choice[j][0]],dataslice[:, choice[j][1]])), bins=30)[1]
        plt.hist(dataslice[:, choice[j][0]], bins=bins, alpha=0.6, histtype='bar', ec='black', color=colour[choice[j][0]], label=labels[choice[j][0]])
        plt.hist(dataslice[:, choice[j][1]], bins=bins, alpha=0.6, histtype='bar', ec='black', color=colour[choice[j][1]], label=labels[choice[j][0]])
        if i==2 and j == 0:
            plt.hist(a6_ideal[:, 9], bins=bins, histtype='bar', ls="dashed", lw=1.5, ec="#1f77b4", zorder=0, fc=(0, 0, 1, 0))
        if i==3 and j == 0:
            plt.hist(a6_real[:, 9], bins=bins, histtype='bar', ls="dashed", lw=1.5, ec="#1f77b4", zorder=0, fc=(0, 0, 1, 0))
        if i==2 and j == 1:
            plt.hist(a6_ideal[:, 9], bins=bins, histtype='bar', ls="dashed", lw=1.5, ec="#1f77b4", zorder=0, fc=(0, 0, 1, 0))
        if i==3 and j == 1:
            plt.hist(a6_real[:, 9], bins=bins, histtype='bar', ls="dashed", lw=1.5, ec="#1f77b4", zorder=0, fc=(0, 0, 1, 0))
        plt.yticks([])

        pval = mannwhitneyu(dataslice[:, choice[j][0]], dataslice[:, choice[j][1]]).pvalue
        uval = mannwhitneyu(dataslice[:, choice[j][0]], dataslice[:, choice[j][1]]).statistic

        if pval < 0.001:
            fig.axes[i*3 + j].text(0.7, 0.85, '***', ma='center', transform = fig.axes[i*3 + j].transAxes)
        elif pval < 0.01:
            fig.axes[i*3 + j].text(0.7, 0.85, '**', ma='center', transform = fig.axes[i*3 + j].transAxes)
        elif pval < 0.05:
            fig.axes[i*3 + j].text(0.7, 0.85, '*', ma='center', transform = fig.axes[i*3 + j].transAxes)
        else:
            fig.axes[i*3 + j].text(0.7, 0.85, 'n.s.', ma='center', transform = fig.axes[i*3 + j].transAxes)
        
        fig.axes[i*3 + j].spines['top'].set_visible(False)
        fig.axes[i*3 + j].spines['right'].set_visible(False)


xoffs = 0.075
plt.text(0.055 - xoffs, 0.73, "Conjunctive", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.098 - xoffs, 0.81, "Ideal", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.098 - xoffs, 0.66, "Doeller", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.055 - xoffs, 0.42, "Repetition-suppression", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.098 - xoffs, 0.55, "Ideal", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.098 - xoffs, 0.41, "Weaker", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.055 - xoffs, 0.15, "Structure-function\n       mapping", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.098 - xoffs, 0.28, "Ideal", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.098 - xoffs, 0.16, "Gu", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)
plt.text(0.09, 0.46, "Probability", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)


fig.axes[-1].set_xlabel("p-l/rand", labelpad=65)
fig.axes[-2].set_xlabel("star/rand", labelpad=65)
fig.axes[-3].set_xlabel("star/p-l", labelpad=65)
# plt.text(0.55, 0.016, "Hexasymmetry (spk/s)", fontsize=settings.fs, transform=plt.gcf().transFigure, rotation=90)


plt.text(0.41, 0.06, "Hexasymmetry (spk/s)", fontsize=settings.fs, transform=plt.gcf().transFigure)

lines1, labels1 = fig.axes[0].get_legend_handles_labels()
lines2, labels2 = fig.axes[1].get_legend_handles_labels()
lines2.reverse()
lines1 = lines1 + lines2
# fig.legend(lines, labels, loc='upper center')

# labels = ["star", "p-l", "rand"]
fig.legend(lines1[:3], labels, loc='center right', bbox_to_anchor=(1.05,0.5), ncol=1, bbox_transform=fig.transFigure)
plt.tight_layout()
plot_fname = os.path.join(settings.loc, "plots", "Figure_pairwise_hist.png")
plt.savefig(
    plot_fname,
    dpi=300
)
plt.close()


###############################################################################
# Percent of paths                                                            #
###############################################################################
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

                pwl_subangs = np.random.choice(pwl_angs, n_pwlangs, replace=False)
                rw_subdirecs = np.random.choice(rw_direcs, n_rwdirs, replace=False)

                pwl_t6[id1, id2] = np.abs(np.sum(np.exp(- 6 * pwl_subangs * 1j))) / (len(pwl_subangs))
                rw_t6[id1, id2] = np.abs(np.sum(np.exp(- 6 * rw_subdirecs * 1j))) / (len(rw_subdirecs))
        
        return pwl_t6, rw_t6


    pwl_t6, rw_t6 = get_t6bars(360, settings.prop_list, pwl_angs)
    with open(t6_fname, "wb") as f:
            pickle.dump((pwl_t6, rw_t6), f)
else:
    with open(t6_fname, "rb") as f:
        pwl_t6, rw_t6 = pickle.load(f)


plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': int(settings.fs - 3)})


# ys[ys<=1e-1] = 1e-1
# ys2[ys2<=1e-1] = 1e-1
# print(ys)
# plt.plot(xs, ys, lw=10)

# path = mpath.Path(np.column_stack([xs, ys]))
# verts = path.interpolated(steps=1).vertices
# xs, ys = verts[:, 0], verts[:, 1]
# zs = np.copy(ys) / np.amax(ys)
zs = (settings.phbins * settings.prop_list / 100)

barwidth=0.25
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
plt.bar(pos[0], np.mean(m6_ideal[:, 7], axis=0) * np.mean(a0_ideal[:, 7], axis=0), width=barwidth, color=[0.67, 0.85, 0.9]) #, edgecolor='k')

plt.bar(pos[1], np.mean(a6_ideal[:, 8], axis=0), width=barwidth, color='b')
plt.bar(pos[1], np.mean(m6_ideal[:, 8], axis=0) * np.mean(a0_ideal[:, 8], axis=0), width=barwidth, color=[0.67, 0.85, 0.9]) #, edgecolor='k')


plt.bar(pos[2], np.mean(a6_real[:, 7], axis=0), width=barwidth, color='b')
plt.bar(pos[2], np.mean(m6_real[:, 7], axis=0) * np.mean(a0_real[:, 7], axis=0), width=barwidth, color=[0.67, 0.85, 0.9])


plt.bar(pos[3], np.mean(a6_real[:, 8], axis=0), width=barwidth, color='b')
plt.bar(pos[3], np.mean(m6_real[:, 8], axis=0) * np.mean(a0_real[:, 8], axis=0), width=barwidth, color=[0.67, 0.85, 0.9])

print(np.amin(ys2), np.amin(ys4))
print(np.mean(m6_ideal[:, 8], axis=0) * np.mean(a0_ideal[:, 8], axis=0), np.mean(m6_real[:, 8], axis=0) * np.mean(a0_real[:, 8], axis=0))

# zs /= np.amax(zs)
colorline(xs, ys, settings.prop_list, cmap=plt.get_cmap('turbo'), linewidth=15, norm=mcol.LogNorm(1, 100))
colorline(xs2, ys2, settings.prop_list, cmap=plt.get_cmap('turbo'), linewidth=15, norm=mcol.LogNorm(1, 100))
colorline(xs3, ys3, settings.prop_list, cmap=plt.get_cmap('turbo'), linewidth=15, norm=mcol.LogNorm(1, 100))
colorline(xs4, ys4, settings.prop_list, cmap=plt.get_cmap('turbo'), linewidth=15, norm=mcol.LogNorm(1, 100))
# colorline(xs, ys, prop_list, cmap=plt.get_cmap('bone'), linewidth=10, norm=plt.Normalize(1, 100))
# colorline(xs2, ys2, prop_list, cmap=plt.get_cmap('bone'), linewidth=10, norm=plt.Normalize(1, 100))
# cmap = plt.get_cmap('bone')
# cbar = plt.colorbar(cmap=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), norm=mcol.LogNorm(1, 100), shrink=0.6, ticks=np.linspace(0, 1, 6, endpoint=True), pad = 0.01)
# cbar.set_ticklabels([0 + 20 * i for i in range(6)])
# cbar.set_label('Percentage of subsampled\nlinear path segments')
# colorline(xs, ys, np.logspace(-1, 0, 100), cmap=plt.get_cmap('brg'), linewidth=15)
plt.yscale("log")
plt.ylabel('Hexasymmetry (spk/s)')
plt.ylim(1e-1, 1e3)
plt.xlim(0.5, 3.3)
plt.xticks([])
plt.xticks(
    pos,
    ['p-l', 'rand', 'p-l', 'rand']
)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.text(0.1, -0.11, "ideal", fontsize=settings.fs, transform=plt.gcf().transFigure)
# plt.text(0.2, -0.11, "Gu", fontsize=settings.fs, transform=plt.gcf().transFigure)
plt.gcf().text(0.275, 0, "ideal", fontsize=settings.fs)
plt.gcf().text(0.625, 0, "Gu", fontsize=settings.fs)

cmap = mpl.cm.turbo
norm = mpl.colors.Normalize(vmin=1, vmax=100)
ax=plt.gca()
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Percentage of subsampled path segments', ax=ax)
plt.yticks([1, 50, 100])

# print(np.amax(ys))
# print(np.amax(ys3))
# print(np.mean(a0_ideal[:, 7]))
# print(np.mean(a0_real[:, 7]))


plt.tight_layout()
plot_fname = os.path.join(settings.loc, "plots", "Figure_percentpaths.png")
plt.savefig(
    plot_fname,
    dpi=300
)
plt.close()