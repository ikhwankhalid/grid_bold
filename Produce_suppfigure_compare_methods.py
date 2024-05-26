import numpy as np
import settings
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from utils.utils import (
    get_hexsym,
    get_pathsym,
    get_glm_hex,
    get_glmbinning_hex,
    circlin_cl,
    divide_dataset,
    create_surrogate_distribution
)
from utils.grid_funcs import (
    traj,
    traj_pwl,
    traj_star,
    gen_offsets,
    gridpop_clustering,
    convert_to_rhombus,
    gridpop_conj,
    gridpop_repsupp
)
from tqdm import tqdm
from scipy.stats import mannwhitneyu, zscore, norm


###############################################################################
# Parameters                                                                  #
###############################################################################
rep = 200
n_jobs = 50
savename = os.path.join(
    settings.sims_dir,
    "traj_conj_clustering",
    "traj_hexsyms_noabs_rollnonrw_allmpath.npy"
)
nlin = int(settings.rmax / settings.speed / settings.dt)


###############################################################################
# Functions                                                                   #
###############################################################################
def get_circlin_pathhex(summed_fr, trajec, rw=True, iters=1000):
    assert summed_fr.ndim == 1, \
        "Input 'summed_fr' must be a one dimensional array"

    pathhex_list = []

    for i in range(iters):
        if rw:
            roll = np.random.randint(0, len(summed_fr))
        else:
            roll = np.random.randint(0, settings.phbins) * nlin

        pathhex_list.append(
            circlin_cl(np.roll(summed_fr, roll), trajec[-1])
        )

    # pathhex = np.mean(pathhex_list)
    pathhex = np.array(pathhex_list)
    return pathhex


def get_4methodhex(trajec, summed_fr, fr_mean, rw=True):
    # Show half data to new method and cl-corr
    groups = divide_dataset(trajec[-1], summed_fr, num_groups=2)
    direc_half, summed_fr_half = groups[0]
    direc_half, summed_fr_half = np.array(direc_half), np.array(summed_fr_half)

    direc = trajec[-1]

    # Create surrogate distribution for cl-corr
    _, summed_fr_surr = create_surrogate_distribution(
        direc, summed_fr
    )

    # Get data
    h_rw = (
        get_hexsym(summed_fr, trajec),
        get_pathsym(trajec[-1]),
        np.mean(summed_fr)
    )
    d_rw = (
        get_glm_hex(trajec[-1], summed_fr)[0],
        0.,
        np.mean(summed_fr[~np.isnan(summed_fr).any()])
    )
    k_rw = (
        get_glmbinning_hex(trajec[-1], summed_fr)[0],
        0.,
        np.mean(summed_fr[~np.isnan(summed_fr).any()])
    )
    m_rw = (
        circlin_cl(summed_fr, trajec[-1]),
        get_circlin_pathhex(summed_fr, trajec, rw),
        # circlin_cl(summed_fr, trajec[-1]),
        np.mean(summed_fr[~np.isnan(summed_fr).any()])
    )

    return h_rw, d_rw, k_rw, m_rw


def get_circlincorr_zvals(trajec, m_hex, fr):
    hexes = np.zeros(fr.shape[0])
    for i in range(len(hexes)):
        hexes[i] = circlin_cl(fr[i, :], trajec[-1])

    zscores = zscore(hexes)

    return zscores


def get_trajhex(trajec, oxr_rand, oyr_rand, oxr_clust, oyr_clust, rw=True):
    # conjunctive
    fr_mean, summed_fr = gridpop_conj(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr_rand,
        oyr=oyr_rand,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )[1::2]
    h_conj, d_conj, k_conj, m_conj = get_4methodhex(
        trajec, summed_fr, fr_mean, rw
    )

    # clustering
    fr_mean, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr_clust,
        oyr=oyr_clust
    )[1::2]
    h_clust, d_clust, k_clust, m_clust = get_4methodhex(
        trajec, summed_fr, fr_mean, rw
    )

    # repsupp
    fr_mean, summed_fr = gridpop_repsupp(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr_rand,
        oyr=oxr_rand
    )[1::2]
    h_repsupp, d_repsupp, k_repsupp, m_repsupp = get_4methodhex(
        trajec, summed_fr, fr_mean, rw
    )

    return (
        h_conj,
        d_conj,
        k_conj,
        m_conj,
        h_clust,
        d_clust,
        k_clust,
        m_clust,
        h_repsupp,
        d_repsupp,
        k_repsupp,
        m_repsupp
    )


def mfunc(i):
    # clustering offsets
    ox_clust, oy_clust = gen_offsets(N=settings.N, kappacl=settings.kappa_si)
    oxr_clust, oyr_clust = convert_to_rhombus(ox_clust, oy_clust)

    # random offsets
    ox_rand, oy_rand = gen_offsets(N=settings.N, kappacl=0.)
    oxr_rand, oyr_rand = convert_to_rhombus(ox_rand, oy_rand)

    # ox_rand, oy_rand = np.meshgrid(
    #     np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False),
    #     np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False)
    # )
    # oxr_rand = ox_rand.reshape(1, -1)[0]
    # oyr_rand = oy_rand.reshape(1, -1)[0]

    ################
    # random walks #
    ################
    # print("rw")
    trajec = traj(settings.dt, settings.tmax)

    (
        h_conj_rw,
        d_conj_rw,
        k_conj_rw,
        m_conj_rw,
        h_clust_rw,
        d_clust_rw,
        k_clust_rw,
        m_clust_rw,
        h_repsupp_rw,
        d_repsupp_rw,
        k_repsupp_rw,
        m_repsupp_rw
    ) = get_trajhex(trajec, oxr_rand, oyr_rand, oxr_clust, oyr_clust)

    ###################
    # star-like walks #
    ###################
    # print("star")
    trajec = traj_star(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )

    (
        h_conj_star,
        d_conj_star,
        k_conj_star,
        m_conj_star,
        h_clust_star,
        d_clust_star,
        k_clust_star,
        m_clust_star,
        h_repsupp_star,
        d_repsupp_star,
        k_repsupp_star,
        m_repsupp_star
    ) = get_trajhex(trajec, oxr_rand, oyr_rand, oxr_clust, oyr_clust, rw=False)

    #####################
    # piece-wise linear #
    #####################
    # print("pwl")
    trajec = traj_pwl(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )

    (
        h_conj_pl,
        d_conj_pl,
        k_conj_pl,
        m_conj_pl,
        h_clust_pl,
        d_clust_pl,
        k_clust_pl,
        m_clust_pl,
        h_repsupp_pl,
        d_repsupp_pl,
        k_repsupp_pl,
        m_repsupp_pl
    ) = get_trajhex(trajec, oxr_rand, oyr_rand, oxr_clust, oyr_clust, rw=False)

    return (
        h_conj_star,
        d_conj_star,
        k_conj_star,
        m_conj_star,
        h_clust_star,
        d_clust_star,
        k_clust_star,
        m_clust_star,
        h_repsupp_star,
        d_repsupp_star,
        k_repsupp_star,
        m_repsupp_star,
        h_conj_pl,
        d_conj_pl,
        k_conj_pl,
        m_conj_pl,
        h_clust_pl,
        d_clust_pl,
        k_clust_pl,
        m_clust_pl,
        h_repsupp_pl,
        d_repsupp_pl,
        k_repsupp_pl,
        m_repsupp_pl,
        h_conj_rw,
        d_conj_rw,
        k_conj_rw,
        m_conj_rw,
        h_clust_rw,
        d_clust_rw,
        k_clust_rw,
        m_clust_rw,
        h_repsupp_rw,
        d_repsupp_rw,
        k_repsupp_rw,
        m_repsupp_rw
    )


###############################################################################
# Run                                                                         #
###############################################################################
os.makedirs(os.path.dirname(savename), exist_ok=True)
if os.path.isfile(savename):
    alldata = np.load(savename, allow_pickle=True)
else:
    alldata = Parallel(
        n_jobs=n_jobs, verbose=100
    )(delayed(mfunc)(i) for i in tqdm(range(rep)))
    alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)

    np.save(savename, alldata)

(
    h_conj_star_0,
    d_conj_star_0,
    k_conj_star_0,
    m_conj_star_0,
    h_clust_star_0,
    d_clust_star_0,
    k_clust_star_0,
    m_clust_star_0,
    h_repsupp_star_0,
    d_repsupp_star_0,
    k_repsupp_star_0,
    m_repsupp_star_0,
    h_conj_pl_0,
    d_conj_pl_0,
    k_conj_pl_0,
    m_conj_pl_0,
    h_clust_pl_0,
    d_clust_pl_0,
    k_clust_pl_0,
    m_clust_pl_0,
    h_repsupp_pl_0,
    d_repsupp_pl_0,
    k_repsupp_pl_0,
    m_repsupp_pl_0,
    h_conj_rw_0,
    d_conj_rw_0,
    k_conj_rw_0,
    m_conj_rw_0,
    h_clust_rw_0,
    d_clust_rw_0,
    k_clust_rw_0,
    m_clust_rw_0,
    h_repsupp_rw_0,
    d_repsupp_rw_0,
    k_repsupp_rw_0,
    m_repsupp_rw_0
) = alldata

d_conj_star = np.array([i[0] for i in d_conj_star_0[0, :]])
d_conj_pl = np.array([i[0] for i in d_conj_pl_0[0, :]])
d_conj_rw = np.array([i[0] for i in d_conj_rw_0[0, :]])

dpath_conj_star = np.array([i for i in d_conj_star_0[1, :]])
dpath_conj_pl = np.array([i for i in d_conj_pl_0[1, :]])
dpath_conj_rw = np.array([i for i in d_conj_rw_0[1, :]])

h_conj_star = np.array([i for i in h_conj_star_0[0, :]])
h_conj_pl = np.array([i for i in h_conj_pl_0[0, :]])
h_conj_rw = np.array([i for i in h_conj_rw_0[0, :]])

hpath_conj_star = np.array([i * j for (i, j) in zip(h_conj_star_0[1, :], h_conj_star_0[2, :])])
hpath_conj_pl = np.array([i * j for (i, j) in zip(h_conj_pl_0[1, :], h_conj_pl_0[2, :])])
hpath_conj_rw = np.array([i * j for (i, j) in zip(h_conj_rw_0[1, :], h_conj_rw_0[2, :])])

k_conj_star = np.array([i for i in k_conj_star_0[0, :]])
k_conj_pl = np.array([i for i in k_conj_pl_0[0, :]])
k_conj_rw = np.array([i for i in k_conj_rw_0[0, :]])

kpath_conj_star = np.array([i for i in k_conj_star_0[1, :]])
kpath_conj_pl = np.array([i for i in k_conj_pl_0[1, :]])
kpath_conj_rw = np.array([i for i in k_conj_rw_0[1, :]])

m_conj_star = np.array([i for i in m_conj_star_0[0, :]])
m_conj_pl = np.array([i for i in m_conj_pl_0[0, :]])
m_conj_rw = np.array([i for i in m_conj_rw_0[0, :]])

mpath_conj_star = np.array([i for i in m_conj_star_0[1, :]])
mpath_conj_pl = np.array([i for i in m_conj_pl_0[1, :]])
mpath_conj_rw = np.array([i for i in m_conj_rw_0[1, :]])

d_clust_star = np.array([i[0] for i in d_clust_star_0[0, :]])
d_clust_pl = np.array([i[0] for i in d_clust_pl_0[0, :]])
d_clust_rw = np.array([i[0] for i in d_clust_rw_0[0, :]])

dpath_clust_star = np.array([i for i in d_clust_star_0[1, :]])
dpath_clust_pl = np.array([i for i in d_clust_pl_0[1, :]])
dpath_clust_rw = np.array([i for i in d_clust_rw_0[1, :]])

h_clust_star = np.array([i for i in h_clust_star_0[0, :]])
h_clust_pl = np.array([i for i in h_clust_pl_0[0, :]])
h_clust_rw = np.array([i for i in h_clust_rw_0[0, :]])

hpath_clust_star = np.array([i * j for (i, j) in zip(h_clust_star_0[1, :], h_clust_star_0[2, :])])
hpath_clust_pl = np.array([i * j for (i, j) in zip(h_clust_pl_0[1, :], h_clust_pl_0[2, :])])
hpath_clust_rw = np.array([i * j for (i, j) in zip(h_clust_rw_0[1, :], h_clust_rw_0[2, :])])

k_clust_star = np.array([i for i in k_clust_star_0[0, :]])
k_clust_pl = np.array([i for i in k_clust_pl_0[0, :]])
k_clust_rw = np.array([i for i in k_clust_rw_0[0, :]])

kpath_clust_star = np.array([i for i in k_clust_star_0[1, :]])
kpath_clust_pl = np.array([i for i in k_clust_pl_0[1, :]])
kpath_clust_rw = np.array([i for i in k_clust_rw_0[1, :]])

m_clust_star = np.array([i for i in m_clust_star_0[0, :]])
m_clust_pl = np.array([i for i in m_clust_pl_0[0, :]])
m_clust_rw = np.array([i for i in m_clust_rw_0[0, :]])

mpath_clust_star = np.array([i for i in m_clust_star_0[1, :]])
mpath_clust_pl = np.array([i for i in m_clust_pl_0[1, :]])
mpath_clust_rw = np.array([i for i in m_clust_rw_0[1, :]])

d_repsupp_star = np.array([i[0] for i in d_repsupp_star_0[0, :]])
d_repsupp_pl = np.array([i[0] for i in d_repsupp_pl_0[0, :]])
d_repsupp_rw = np.array([i[0] for i in d_repsupp_rw_0[0, :]])

dpath_repsupp_star = np.array([i for i in d_repsupp_star_0[1, :]])
dpath_repsupp_pl = np.array([i for i in d_repsupp_pl_0[1, :]])
dpath_repsupp_rw = np.array([i for i in d_repsupp_rw_0[1, :]])

h_repsupp_star = np.array([i for i in h_repsupp_star_0[0, :]])
h_repsupp_pl = np.array([i for i in h_repsupp_pl_0[0, :]])
h_repsupp_rw = np.array([i for i in h_repsupp_rw_0[0, :]])

hpath_repsupp_star = np.array([i * j for (i, j) in zip(h_repsupp_star_0[1, :], h_repsupp_star_0[2, :])])
hpath_repsupp_pl = np.array([i * j for (i, j) in zip(h_repsupp_pl_0[1, :], h_repsupp_pl_0[2, :])])
hpath_repsupp_rw = np.array([i * j for (i, j) in zip(h_repsupp_rw_0[1, :], h_repsupp_rw_0[2, :])])

print(hpath_repsupp_rw.shape)

k_repsupp_star = np.array([i for i in k_repsupp_star_0[0, :]])
k_repsupp_pl = np.array([i for i in k_repsupp_pl_0[0, :]])
k_repsupp_rw = np.array([i for i in k_repsupp_rw_0[0, :]])

kpath_repsupp_star = np.array([i for i in k_repsupp_star_0[1, :]])
kpath_repsupp_pl = np.array([i for i in k_repsupp_pl_0[1, :]])
kpath_repsupp_rw = np.array([i for i in k_repsupp_rw_0[1, :]])

m_repsupp_star = np.array([i for i in m_repsupp_star_0[0, :]])
m_repsupp_pl = np.array([i for i in m_repsupp_pl_0[0, :]])
m_repsupp_rw = np.array([i for i in m_repsupp_rw_0[0, :]])

mpath_repsupp_star = np.array([i for i in m_repsupp_star_0[1, :]])
mpath_repsupp_pl = np.array([i for i in m_repsupp_pl_0[1, :]])
mpath_repsupp_rw = np.array([i for i in m_repsupp_rw_0[1, :]])


def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))


###############################################################################
viol_pos = np.array([0.7, 1.8, 2.9, 4.2, 5.3, 6.4, 7.7, 8.8, 9.9])
viol_pos += np.array([i*0.1 for i in range(len(viol_pos))])
spaces = 0.25
# viol
labels = []

plt.rcParams.update({'font.size': settings.fs*0.95})
# Create violin plot
fig, ax = plt.subplots(figsize=(12, 5))
# conjunctive
add_label(
    ax.violinplot(
        [
            d_conj_star/2,
            d_conj_pl/2,
            d_conj_rw/2,
            d_repsupp_star/2,
            d_repsupp_pl/2,
            d_repsupp_rw/2,
            d_clust_star/2,
            d_clust_pl/2,
            d_clust_rw/2
        ],
        showmeans=False,
        showmedians=True,
        positions=viol_pos,
        widths=0.25
    ),
    label="GLM"
)
add_label(
    ax.violinplot(
        [
            k_conj_star[~np.isnan(k_conj_star)]/2,
            k_conj_pl[~np.isnan(k_conj_pl)]/2,
            k_conj_rw[~np.isnan(k_conj_rw)]/2,
            k_repsupp_star[~np.isnan(k_repsupp_star)]/2,
            k_repsupp_pl[~np.isnan(k_repsupp_pl)]/2,
            k_repsupp_rw[~np.isnan(k_repsupp_rw)]/2,
            k_clust_star[~np.isnan(k_clust_star)]/2,
            k_clust_pl[~np.isnan(k_clust_pl)]/2,
            k_clust_rw[~np.isnan(k_clust_rw)]/2
        ],
        showmeans=False,
        showmedians=True,
        positions=viol_pos + spaces,
        widths=0.25
    ),
    label="GLM (binning)"
)
add_label(
    ax.violinplot(
        [
            h_conj_star,
            h_conj_pl,
            h_conj_rw,
            h_repsupp_star,
            h_repsupp_pl,
            h_repsupp_rw,
            h_clust_star,
            h_clust_pl,
            h_clust_rw
        ],
        showmeans=False,
        showmedians=True,
        positions=viol_pos + 2 * spaces,
        widths=0.25
    ),
    label="New method"
)
# add_label(
#     ax.violinplot(
#         [
#             m_conj_star[~np.isnan(m_conj_star)],
#             m_conj_pl[~np.isnan(m_conj_pl)],
#             m_conj_rw[~np.isnan(m_conj_rw)],
#             m_repsupp_star[~np.isnan(m_repsupp_star)],
#             m_repsupp_pl[~np.isnan(m_repsupp_pl)],
#             m_repsupp_rw[~np.isnan(m_repsupp_rw)],
#             m_clust_star[~np.isnan(m_clust_star)],
#             m_clust_pl[~np.isnan(m_clust_pl)],
#             m_clust_rw[~np.isnan(m_clust_rw)]
#         ],
#         showmeans=False,
#         showmedians=True,
#         positions=viol_pos + 3 * spaces,
#         widths=0.25
#     ),
#     label="CL-corr"
# )

# Significance
# add asterisks for statistical significance
alldat_fr = [
    d_conj_star/2,
    d_conj_pl/2,
    d_conj_rw/2,
    d_repsupp_star/2,
    d_repsupp_pl/2,
    d_repsupp_rw/2,
    d_clust_star/2,
    d_clust_pl/2,
    d_clust_rw/2,
    k_conj_star[~np.isnan(k_conj_star)]/2,
    k_conj_pl[~np.isnan(k_conj_pl)]/2,
    k_conj_rw[~np.isnan(k_conj_rw)]/2,
    k_repsupp_star[~np.isnan(k_repsupp_star)]/2,
    k_repsupp_pl[~np.isnan(k_repsupp_pl)]/2,
    k_repsupp_rw[~np.isnan(k_repsupp_rw)]/2,
    k_clust_star[~np.isnan(k_clust_star)]/2,
    k_clust_pl[~np.isnan(k_clust_pl)]/2,
    k_clust_rw[~np.isnan(k_clust_rw)]/2,
    h_conj_star,
    h_conj_pl,
    h_conj_rw,
    h_repsupp_star,
    h_repsupp_pl,
    h_repsupp_rw,
    h_clust_star,
    h_clust_pl,
    h_clust_rw,
    m_conj_star[~np.isnan(m_conj_star)],
    m_conj_pl[~np.isnan(m_conj_pl)],
    m_conj_rw[~np.isnan(m_conj_rw)],
    m_repsupp_star[~np.isnan(m_repsupp_star)],
    m_repsupp_pl[~np.isnan(m_repsupp_pl)],
    m_repsupp_rw[~np.isnan(m_repsupp_rw)],
    m_clust_star[~np.isnan(m_clust_star)],
    m_clust_pl[~np.isnan(m_clust_pl)],
    m_clust_rw[~np.isnan(m_clust_rw)]
]
alldat_path = [
    dpath_conj_star,
    dpath_conj_pl,
    dpath_conj_rw,
    dpath_repsupp_star,
    dpath_repsupp_pl,
    dpath_repsupp_rw,
    dpath_clust_star,
    dpath_clust_pl,
    dpath_clust_rw,
    kpath_conj_star,
    kpath_conj_pl,
    kpath_conj_rw,
    kpath_repsupp_star,
    kpath_repsupp_pl,
    kpath_repsupp_rw,
    kpath_clust_star,
    kpath_clust_pl,
    kpath_clust_rw,
    hpath_conj_star,
    hpath_conj_pl,
    hpath_conj_rw,
    hpath_repsupp_star,
    hpath_repsupp_pl,
    hpath_repsupp_rw,
    hpath_clust_star,
    hpath_clust_pl,
    hpath_clust_rw,
    mpath_conj_star,
    mpath_conj_pl,
    mpath_conj_rw,
    mpath_repsupp_star,
    mpath_repsupp_pl,
    mpath_repsupp_rw,
    mpath_clust_star,
    mpath_clust_pl,
    mpath_clust_rw
]

print(alldat_path[27].shape)


def zscore(x, mean, std):
    return (x - mean) / std


def p_value(z_score):
    return norm.sf(abs(z_score))  #two-sided


pvals = np.zeros(len(alldat_fr))
z_scores = np.zeros((9, 200))
for i in range(len(alldat_fr)):
    if i > 26:
        zscores_temp = np.zeros(rep)
        for j in range(rep):
            zscores_temp[j] = zscore(
                alldat_fr[i][j],
                np.mean(alldat_path[i][j, :]),
                np.std(alldat_path[i][j, :])
            )
            z_scores[i - 27, j] = zscores_temp[j]
        pvals[i] = mannwhitneyu(zscores_temp, 0).pvalue
    else:
        pvals[i] = mannwhitneyu(alldat_fr[i], alldat_path[i]).pvalue
print("pval: ", pvals)
pos = np.tile(viol_pos, 4)
for i in range(4):
    pos[9*i:9 + 9*i] += spaces * i

print(pos)
plt.scatter(
    pos[18:27],
    np.median(
        np.array(
            [
                hpath_conj_star,
                hpath_conj_pl,
                hpath_conj_rw,
                hpath_repsupp_star,
                hpath_repsupp_pl,
                hpath_repsupp_rw,
                hpath_clust_star,
                hpath_clust_pl,
                hpath_clust_rw
            ]
        ),
        axis=1
    ),
    color="black",
    marker="s",
    zorder=10
)

plt.scatter(
    pos[27:],
    np.median(
        np.array(
            [
                mpath_conj_star.flatten(),
                mpath_conj_pl.flatten(),
                mpath_conj_rw.flatten(),
                mpath_repsupp_star.flatten(),
                mpath_repsupp_pl.flatten(),
                mpath_repsupp_rw.flatten(),
                mpath_clust_star.flatten(),
                mpath_clust_pl.flatten(),
                mpath_clust_rw.flatten()
            ]
        ),
        axis=1
    ),
    color="black",
    marker="*",
    s=60.,
    zorder=10
)

# add_label(
#     ax.violinplot(
#         [
#             mpath_conj_star.flatten(),
#             mpath_conj_pl.flatten(),
#             mpath_conj_rw.flatten(),
#             mpath_repsupp_star.flatten(),
#             mpath_repsupp_pl.flatten(),
#             mpath_repsupp_rw.flatten(),
#             mpath_clust_star.flatten(),
#             mpath_clust_pl.flatten(),
#             mpath_clust_rw.flatten()
#         ],
#         showmeans=False,
#         showmedians=True,
#         positions=viol_pos + 3.5 * spaces,
#         widths=0.25
#     ),
#     label="CL-corr path"
# )

# print(len(z_scores), len(viol_pos))

# add_label(
#     ax.violinplot(
#         [
#             z_scores[0, :],
#             z_scores[1, :],
#             z_scores[2, :],
#             z_scores[3, :],
#             z_scores[4, :],
#             z_scores[5, :],
#             z_scores[6, :],
#             z_scores[7, :],
#             z_scores[8, :]
#         ],
#         showmeans=False,
#         showmedians=True,
#         positions=viol_pos + 3 * spaces,
#         widths=0.25
#     ),
#     label="CL-corr (Z-scores)"
# )

add_label(
    ax.violinplot(
        [
            m_conj_star[~np.isnan(m_conj_star)],
            m_conj_pl[~np.isnan(m_conj_pl)],
            m_conj_rw[~np.isnan(m_conj_rw)],
            m_repsupp_star[~np.isnan(m_repsupp_star)],
            m_repsupp_pl[~np.isnan(m_repsupp_pl)],
            m_repsupp_rw[~np.isnan(m_repsupp_rw)],
            m_clust_star[~np.isnan(m_clust_star)],
            m_clust_pl[~np.isnan(m_clust_pl)],
            m_clust_rw[~np.isnan(m_clust_rw)]
        ],
        showmeans=False,
        showmedians=True,
        positions=viol_pos + 3 * spaces,
        widths=0.25
    ),
    label="CL-corr"
)

for i in range(len(alldat_fr)):
    # data = z_scores[i - 27] if i >= 27 else alldat_fr[i]
    data = alldat_fr[i]
    if i == 25 or i == 26:
        plt.text(
            pos[i],
            1.5*np.max(data),
            '(***)',
            horizontalalignment='center',
            fontsize=0.8*settings.fs
        )
    elif i == 23:
        plt.text(
            pos[i],
            1.5*np.max(data),
            'n.s.',
            horizontalalignment='center',
            fontsize=0.8*settings.fs
        )
    elif pvals[i] < 0.001:
        plt.text(
            pos[i],
            1.5*np.max(data),
            '***',
            horizontalalignment='center',
            fontsize=0.8*settings.fs
        )
    elif pvals[i] < 0.01:
        plt.text(
            pos[i],
            1.5*np.max(data),
            '**',
            horizontalalignment='center',
            fontsize=0.8*settings.fs
        )
    elif pvals[i] < 0.05:
        plt.text(
            pos[i],
            1.5*np.max(data),
            '*',
            horizontalalignment='center',
            fontsize=0.8*settings.fs
        )
    else:
        plt.text(
            pos[i],
            1.5*np.max(data),
            'n.s.',
            horizontalalignment='center',
            fontsize=0.8*settings.fs
        )

# Customize plot labels
ax.set_xticks(viol_pos + 0.3)
ax.set_xticklabels(
    [
        "star",
        "p-l",
        "rand",
        "star",
        "p-l",
        "rand",
        "star",
        "p-l",
        "rand"
    ],
    fontsize=settings.fs
)
pos_ordered = np.sort(pos)
plt.vlines(
    (pos_ordered[11] + pos_ordered[12])/2,
    -1e2,
    1e4,
    color="black",
    linewidth=0.5
)
plt.vlines(
    (pos_ordered[23] + pos_ordered[24])/2,
    -1e2,
    1e4,
    color="black",
    linewidth=0.5
)
plt.yscale('symlog', linthresh=10.)
plt.text(
    0.24,
    0.04,
    "Conjunctive",
    fontsize=settings.fs,
    transform=plt.gcf().transFigure,
    ha='center'
)
plt.text(
    0.530,
    0.01,
    "Repetition\nsuppression",
    fontsize=settings.fs,
    transform=plt.gcf().transFigure,
    ha='center'
)
plt.text(
    0.83,
    0.01,
    "Structure-function\nmapping",
    fontsize=settings.fs,
    transform=plt.gcf().transFigure,
    ha='center'
)
ax.set_ylabel("Hexasymmetry (spk/s)", fontsize=settings.fs)
# ax.set_title(f"Ideal parameters, {rep} realisations")
plt.hlines(0., 0, 12, linestyle="--", linewidth=1.5, color="black", zorder=0)
plt.legend(*zip(*labels))
plt.ylim(-100, 9000)
plt.xlim(0.3, np.amax(viol_pos) + 1.)
plt.tight_layout()

print(
    f"GLM conj star:{pvals[0]}"
)
print(
    f"GLM conj pl:{pvals[1]}"
)
print(
    f"GLM conj rw:{pvals[2]}"
)

print(
    f"GLM repsupp star:{pvals[3]}"
)
print(
    f"GLM repsupp pl:{pvals[4]}"
)
print(
    f"GLM repsupp rw:{pvals[5]}"
)

print(
    f"GLM clust star:{pvals[6]}"
)
print(
    f"GLM clust pl:{pvals[7]}"
)
print(
    f"GLM clust rw:{pvals[8]}"
)

############

print(
    f"GLM (bin) conj star:{pvals[9]}"
)
print(
    f"GLM (bin) conj pl:{pvals[10]}"
)
print(
    f"GLM (bin) conj rw:{pvals[11]}"
)

print(
    f"GLM (bin) repsupp star:{pvals[12]}"
)
print(
    f"GLM (bin) repsupp pl:{pvals[13]}"
)
print(
    f"GLM (bin) repsupp rw:{pvals[14]}"
)

print(
    f"GLM (bin) clust star:{pvals[15]}"
)
print(
    f"GLM (bin) clust pl:{pvals[16]}"
)
print(
    f"GLM (bin) clust rw:{pvals[17]}"
)

###########

print(
    f"New conj star:{pvals[18]:.4f}"
)
print(
    f"New conj pl:{pvals[19]:.4f}"
)
print(
    f"New conj rw:{pvals[20]:.4f}"
)

print(
    f"New repsupp star:{pvals[21]:.4f}"
)
print(
    f"New repsupp pl:{pvals[22]:.4f}"
)
print(
    f"New repsupp rw:{pvals[23]:.4f}"
)

print(
    f"New clust star:{pvals[24]:.4f}"
)
print(
    f"New clust pl:{pvals[25]:.4f}"
)
print(
    f"New clust rw:{pvals[26]:.4f}"
)

#################

print(
    f"CL-corr conj star:{pvals[27]:.3f}"
)
print(
    f"CL-corr conj pl:{pvals[28]:.3f}"
)
print(
    f"CL-corr conj rw:{pvals[29]:.3f}"
)

print(
    f"CL-corr repsupp star:{pvals[30]:.3f}"
)
print(
    f"CL-corr repsupp pl:{pvals[31]:.3f}"
)
print(
    f"CL-corr repsupp rw:{pvals[32]:.3f}"
)

print(
    f"CL-corr clust star:{pvals[33]:.3f}"
)
print(
    f"CL-corr clust pl:{pvals[34]:.3f}"
)
print(
    f"CL-corr clust rw:{pvals[35]:.3f}"
)

# Show the plot
plt.show()
plt.close()
