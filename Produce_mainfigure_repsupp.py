import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import settings
import os
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.utils import (
    grid_meanfr,
    grid_2d,
    get_hexsym,
    get_pathsym,
    convert_to_rhombus,
    adap_euler,
    ax_pos
)
from utils.grid_funcs import (
    traj, gridpop_repsupp, traj_pwl, traj_star2 as traj_star, gen_offsets
)
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
matplotlib.use('Agg')


##############################################################################
# Parameters                                                                 #
##############################################################################
N = settings.N
# imax = settings.imax
imax = 10
fs = settings.fs
extend = False
trajc = "w"


# trajectory parameters
bins = settings.bins
rmax = settings.rmax
part = 4000

# yticksnums = [750, 800, 850, 900]
yticksnums = [750, 900]


pwl_traj_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "pwl_traj.pkl"
)
rw_traj_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "rw_traj.pkl"
)


plt.rcParams['pcolor.shading'] = 'nearest'
fig = plt.figure(figsize=(22, 9.5))
spec = fig.add_gridspec(
    ncols=6,
    nrows=6,
    width_ratios=[1.5, 1, 1, 1, 1, 0.5],
    height_ratios=[1, 1, 1, 1, 1, 1]
)


# plot grid and direction of movement
ax_prefdir = fig.add_subplot(spec[0:2, 0])
X, Y = np.meshgrid(np.linspace(-50, 50, 1000), np.linspace(-50, 50, 1000))
gr = settings.amax*grid_2d(
    X, Y, grsc=settings.grsc, angle=0, offs=np.array([0, 0])
)
pcolor_prefdir = plt.pcolor(X, Y, gr)  # vmin=0,vmax=20)
plt.arrow(
    0,
    0,
    1.5*15,
    1.5*np.sqrt(3)/2*30,
    color='pink',
    length_includes_head=True,
    width=4,
    head_width=9
)
plt.arrow(
    0,
    0,
    1.5*np.sin(np.pi/3)*30,
    1.5*np.cos(np.pi/3)*30,
    color='darkgray',
    length_includes_head=True,
    width=4,
    head_width=9
)
ax_prefdir.set_aspect('equal')
plt.xlabel('x', fontsize=settings.fs)
plt.ylabel('y', fontsize=settings.fs)
plt.xticks([])
plt.yticks([])


# ox,oy = np.meshgrid(
#         np.linspace(0,1,int(np.sqrt(N)),endpoint=False),
#         np.linspace(0,1,int(np.sqrt(N)),endpoint=False)
#     )
# ox = ox.reshape(1,-1)[0]
# oy = oy.reshape(1,-1)[0]
# ox, oy = gen_offsets(N=settings.N, kappacl=0.)
# ox, oy = gen_offsets(N=N, kappacl=0.)
ox, oy = np.zeros(settings.N), np.zeros(settings.N)
oxr, oyr = convert_to_rhombus(ox, oy)


# fr over time plot
tt = np.linspace(0, rmax/settings.speed, bins)
r, phi, indoff = np.meshgrid(
    np.linspace(0, rmax, bins),
    np.linspace(0, 2 * np.pi, settings.phbins, endpoint=False),
    np.arange(len(ox))
)
stargrids_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "stargrids.pkl"
)
if not os.path.isfile(stargrids_fname):
    # star_offs = np.random.rand(2)
    star_offs = [0, 0]
    offxr, offyr = convert_to_rhombus(star_offs[0], star_offs[1])
    X, Y = r*np.cos(phi), r*np.sin(phi)
    grids = settings.amax * grid_meanfr(X, Y, offs=(oxr, oyr))
    grids2 = grids.copy()
    for idir in range(settings.phbins):
        for ic in range(N):
            v = adap_euler(
                grids[idir, :, ic], tt, settings.tau_rep, settings.w_rep
            )
            grids[idir, :, ic] = v
    os.makedirs(os.path.dirname(stargrids_fname), exist_ok=True)
    with open(stargrids_fname, 'wb') as f:
        pickle.dump((grids, grids2), f)
else:
    with open(stargrids_fname, 'rb') as f:
        grids, grids2 = pickle.load(f)


# plt.subplot(3, 6, (2,3))
ax_fr = fig.add_subplot(spec[0:2, 1:3])
ax_fr.spines['top'].set_visible(False)
ax_fr.spines['right'].set_visible(False)
plt.plot(tt, grids2[30, :, 0], color='darkgray', linestyle='-', lw=3)
plt.plot(tt, grids[30, :, 0], color='darkgray', linestyle='--', lw=3)
plt.plot(tt, grids2[60, :, 0], color='pink', linestyle='-', lw=3)
plt.plot(tt, grids[60, :, 0], color='pink', linestyle='--', lw=3)
plt.xlabel('Time (s)', fontsize=settings.fs)
plt.ylabel('Firing rate\n(spk/s)', fontsize=settings.fs)
plt.xticks(fontsize=settings.fs)
plt.yticks([0, 8], fontsize=settings.fs)
plt.xticks([0, 5, 10], fontsize=settings.fs)
plt.xlim([0, 11])

print("Ideal params:")
print(
    "Attenuation aligned:",
    (np.amax(grids2[60, 333:, 0]) - np.amax(grids[60, 333:, 0]))
    / np.amax(grids2[60, 333:, 0]) * 100
)
print(
    "Attenuation misaligned:",
    (np.amax(grids2[30, 333:, 0]) - np.amax(grids[30, 333:, 0]))
    / np.amax(grids2[30, 333:, 0]) * 100
)

##############################################################################
# Star-like walks                                                            #
##############################################################################
# starlike plot
print("Starlike")
# ax2 = plt.subplot(3, 6, (4,5))
ax_star = fig.add_subplot(spec[0:2, 3:5])
ax_star.spines['top'].set_visible(False)
ax_star.spines['right'].set_visible(False)


# star-like walk with reset
grsstar_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "grsstar.pkl"
)
mfrstar_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "mfrstar.pkl"
)


def star_h(i):
    ox, oy = gen_offsets(N=N, kappacl=0.)
    oxr, oyr = convert_to_rhombus(ox, oy)
    X, Y = r*np.cos(phi), r*np.sin(phi)
    grids = settings.amax * grid_meanfr(X, Y, offs=(oxr, oyr))
    grids2 = grids.copy()
    for idir in range(settings.phbins):
        for ic in range(N):
            v = adap_euler(
                grids[idir, :, ic], tt, settings.tau_rep, settings.w_rep
            )
            grids[idir, :, ic] = v
    meanfr = np.sum(np.sum(grids, axis=1) / bins, axis=1)
    gr2 = np.sum(grids, axis=2)
    gr60 = np.abs(np.sum(gr2 * np.exp(-6j*phi[:, :, 0])))/np.size(gr2)

    return gr60, meanfr


if not os.path.isfile(grsstar_fname) or not os.path.isfile(mfrstar_fname):
    alldata = Parallel(
        n_jobs=4, verbose=100
    )(delayed(star_h)(i) for i in tqdm(range(imax)))
    alldata = np.array(alldata)
    alldata = np.moveaxis(np.moveaxis(alldata, 1, 0), 1, -1)
    gr60_reset, meanfr_reset = alldata
    meanfr_reset = np.hstack(meanfr_reset).reshape(imax, settings.phbins)

    os.makedirs(os.path.dirname(grsstar_fname), exist_ok=True)
    with open(grsstar_fname, 'wb') as f:
        pickle.dump(gr60_reset, f)
    with open(mfrstar_fname, 'wb') as f:
        pickle.dump(meanfr_reset, f)
else:
    with open(grsstar_fname, 'rb') as f:
        gr60_reset = pickle.load(f)
    with open(mfrstar_fname, 'rb') as f:
        meanfr_reset = pickle.load(f)


plt.plot(np.linspace(0, 360, settings.phbins), meanfr_reset[0, :], 'k')
plt.ylim([750, 900])
plt.xticks([0, 60, 120, 180, 240, 300, 360], [])
plt.yticks(yticksnums, fontsize=settings.fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=settings.fs)


# star-like walk with no reset
grsstar_noreset_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "grsstar_noreset.pkl"
)
mfrstar_noreset_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "mfrstar_noreset.pkl"
)


def star_noreset_h(i):
    ox, oy = gen_offsets(N=N, kappacl=0.)
    oxr, oyr = convert_to_rhombus(ox, oy)
    trajec = traj_star(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )
    direc_binned, meanfr, _, summedfr = gridpop_repsupp(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    gr60 = get_hexsym(summedfr, trajec)
    meanfr = np.zeros(settings.phbins)
    angles = np.linspace(- np.pi, np.pi, settings.phbins, endpoint=False)
    for i, dir in enumerate(angles):
        dir_idx = int(np.round(np.rad2deg(dir)))
        meanfr[dir_idx] = np.sum(summedfr[trajec[-1]==dir]) / len(summedfr[trajec[-1]==dir])

    return gr60, meanfr


if not os.path.isfile(grsstar_noreset_fname) or not os.path.isfile(mfrstar_noreset_fname):
    alldata = Parallel(
        n_jobs=4, verbose=100)(delayed(star_noreset_h)(i) for i in tqdm(range(imax))
    )
    alldata = np.array(alldata)
    alldata = np.moveaxis(np.moveaxis(alldata, 1, 0), 1, -1)
    gr60_star_noreset, meanfr_star_noreset = alldata
    meanfr_star_noreset = np.hstack(
        meanfr_star_noreset
    ).reshape(imax, settings.phbins)

    # trajec = traj_star(
    #     settings.phbins,
    #     settings.rmax,
    #     settings.dt,
    #     sp=settings.speed
    # )
    # direc_binned, meanfr_star_noreset, _, summedfr = gridpop_repsupp(
    #     settings.N,
    #     grsc=settings.grsc,
    #     phbins=settings.phbins,
    #     traj=trajec,
    #     oxr=oxr,
    #     oyr=oyr
    # )
    # gr60_star = get_hexsym(summedfr, trajec)
    # meanfr_star_noreset = np.zeros(settings.phbins)
    # angles = np.linspace(- np.pi, np.pi, settings.phbins, endpoint=False)
    # for i, dir in enumerate(angles):
    #     dir_idx = int(np.round(np.rad2deg(dir)))
    #     meanfr_star_noreset[dir_idx] = np.sum(summedfr[trajec[-1]==dir]) / len(summedfr[trajec[-1]==dir])

    os.makedirs(os.path.dirname(grsstar_noreset_fname), exist_ok=True)
    with open(grsstar_noreset_fname, 'wb') as f:
        pickle.dump(gr60_star_noreset, f)
    with open(mfrstar_noreset_fname, 'wb') as f:
        pickle.dump(meanfr_star_noreset, f)
else:
    with open(grsstar_noreset_fname, 'rb') as f:
        gr60_star_noreset = pickle.load(f)
    with open(mfrstar_noreset_fname, 'rb') as f:
        meanfr_star_noreset = pickle.load(f)


trajec = traj_star(
    settings.phbins,
    settings.rmax,
    settings.dt,
    sp=settings.speed
)


plt.plot(
    np.linspace(0, 360, settings.phbins),
    meanfr_star_noreset[np.random.randint(0, imax), :],
    'k--'
)
plt.text(
    0.62,
    0.89,
    'H = ' + str(np.round(np.median(gr60_reset, axis=-1), 1))
    + f" spk/s, ({np.round(np.median(gr60_star_noreset), 1)} spk/s)",
    fontsize=settings.fs,
    transform=plt.gcf().transFigure
)


# star trajectory plot
ax_star_traj = fig.add_subplot(spec[0:2, 5])
plt.plot([0, 0], [-90, 90], trajc, linewidth=1.5)
plt.plot([-90, 90], [0, 0], trajc, linewidth=1.5)
plt.plot(
    [-1/np.sqrt(2) * 90, 1/np.sqrt(2) * 90],
    [-1/np.sqrt(2) * 90, 1/np.sqrt(2) * 90],
    trajc,
    linewidth=1.5
)
plt.plot(
    [-1/np.sqrt(2) * 90, 1/np.sqrt(2) * 90],
    [1/np.sqrt(2) * 90, -1/np.sqrt(2) * 90],
    trajc,
    linewidth=1.5
)
ax_star_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
X_bgr, Y_bgr, _ = np.meshgrid(
    np.linspace(-120, 120, 1000),
    np.linspace(-120, +120, 1000),
    0
)
gr_bgr = grid_meanfr(X_bgr, Y_bgr, grsc=30, angle=0, offs=np.array([0, 0]))
ax_star_traj.yaxis.set_ticks_position("right")
ax_star_traj.pcolor(
    X_bgr[:, :, 0],
    Y_bgr[:, :, 0],
    gr_bgr[:, :, 0],
    shading='auto'
)
plt.axis("square")


bar_dist = 4 * settings.grsc
bar_ang = -120
bar_len = 2 * settings.grsc
bar_off = settings.grsc
plt.plot(
    [
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_off,
        bar_dist * np.cos(2*np.pi*bar_ang/360) - bar_len + bar_off,
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len + bar_off
    ],
    [
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360)
    ],
    color="red",
    lw=3
)


###############################################################################
# piecewise linear plot                                                       #
###############################################################################
ox, oy = gen_offsets(N=N, kappacl=0.)
oxr, oyr = convert_to_rhombus(ox, oy)
print("Piecewise linear")
# piecewise linear firing rate subplot
ax_pw = fig.add_subplot(spec[2:4, 3:5])
ax_pw.spines['top'].set_visible(False)
ax_pw.spines['right'].set_visible(False)
trajec = traj_pwl(
    settings.phbins,
    settings.rmax,
    settings.dt,
    sp=settings.speed
)


mfrpwl_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "mfrpwl.pkl"
)
if not os.path.isfile(mfrpwl_fname):
    direc_binned, meanfr_pwl, _, summed_fr = gridpop_repsupp(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    # meanfr_pwl = np.zeros(settings.phbins)
    # angles = np.linspace(- np.pi, np.pi, settings.phbins, endpoint=False)
    # for i, dir in enumerate(angles):
    #     dir_idx = int(np.round(np.rad2deg(dir)))
    #     meanfr_pwl[dir_idx] = np.sum(
    #         summed_fr[trajec[-1]==dir]) / len(summed_fr[trajec[-1]==dir]
    #     )
    with open(mfrpwl_fname, 'wb') as f:
        pickle.dump(meanfr_pwl, f)
else:
    with open(mfrpwl_fname, 'rb') as f:
        meanfr_pwl = pickle.load(f)
plt.plot(np.linspace(0, 360, settings.phbins, endpoint=False), meanfr_pwl, 'k')
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=settings.fs)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [])
plt.yticks(yticksnums, fontsize=settings.fs)
plt.ylim([750, 900])


print("Hex for pwl")
# pwl hexasymmetry
grs_pl = np.array([])
grs_path_pl = np.array([])
grspl_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "grspl.pkl"
)

i = 0
if os.path.exists(grspl_fname):
    with open(grspl_fname, 'rb') as f:
        grs_pl = pickle.load(f)
        if extend:
            while i < imax:
                print(i)
                ox, oy = gen_offsets(N=settings.N)
                oxr, oyr = convert_to_rhombus(ox, oy)
                trajec = traj_pwl(
                    settings.phbins_pwl,
                    settings.rmax,
                    settings.dt,
                    sp=settings.speed
                )
                direc_binned, meanfr, _, summed_fr = gridpop_repsupp(
                    settings.N,
                    grsc=settings.grsc,
                    phbins=settings.phbins,
                    traj=trajec,
                    oxr=oxr,
                    oyr=oyr
                )
                gr60_pl = get_hexsym(summed_fr, trajec)
                gr60_path_pl = get_pathsym(trajec)
                grs_pl = np.hstack((grs_pl, gr60_pl))
                grs_path_pl = np.hstack((grs_path_pl, gr60_path_pl))
                i += 1

            with open(grspl_fname, 'wb') as f:
                pickle.dump(grs_pl, f)
else:
    while i < imax:
        print(i)
        ox, oy = gen_offsets(N=settings.N)
        oxr, oyr = convert_to_rhombus(ox, oy)
        trajec = traj_pwl(
            settings.phbins_pwl,
            settings.rmax,
            settings.dt,
            sp=settings.speed
        )
        direc_binned, meanfr, _, summed_fr = gridpop_repsupp(
            settings.N,
            grsc=settings.grsc,
            phbins=settings.phbins,
            traj=trajec,
            oxr=oxr,
            oyr=oyr
        )
        gr60_pl = get_hexsym(summed_fr, trajec)
        gr60_path_pl = get_pathsym(trajec)
        grs_pl = np.hstack((grs_pl, gr60_pl))
        grs_path_pl = np.hstack((grs_path_pl, gr60_path_pl))
        i += 1

    with open(grspl_fname, 'wb') as f:
        pickle.dump(grs_pl, f)


grs_pl = np.median(grs_pl)
plt.text(
    0.48,
    0.9,
    'H = '+str(np.round(grs_pl, 1))+" spk/s",
    fontsize=settings.fs,
    transform=ax_pw.transAxes
)


# pwl trajectory subplot
pwl_traj_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "pwl_traj.pkl"
)
ax_pwl_traj = fig.add_subplot(spec[2:4, 5])
# with open(pwl_traj_fname, 'rb') as f:
#         trajec = pickle.load(f)
# trajec = traj_pwl(
#     settings.phbins,
#     settings.rmax,
#     settings.dt,
#     sp=settings.speed
# )
if not os.path.isfile(pwl_traj_fname):
    trajec = traj_pwl(
        settings.phbins_pwl,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )
    os.makedirs(os.path.dirname(pwl_traj_fname), exist_ok=True)
    with open(pwl_traj_fname, 'wb') as f:
        pickle.dump(trajec, f)
else:
    with open(pwl_traj_fname, 'rb') as f:
        trajec = pickle.load(f)

plt.plot(
    trajec[1][:int(part*2)], trajec[2][:int(part*2)], trajc, linewidth=1.5
)
ax_pwl_traj.set_aspect('equal', adjustable='box')
plt.axis('square')
plt.xticks([])
plt.yticks([])


bar_dist = 4 * settings.grsc
bar_ang = 60
bar_off = 5*settings.grsc
plt.plot(
    [
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_off,
        bar_dist * np.cos(2*np.pi*bar_ang/360) - bar_len + bar_off,
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len + bar_off
    ],
    [
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360)
    ],
    color="red",
    lw=3
)


X_bgr, Y_bgr, _ = np.meshgrid(
    np.linspace(-2000, 2000, 4000),
    np.linspace(-2000, 2000, 4000),
    0
)
gr_bgr = grid_meanfr(
    X_bgr,
    Y_bgr,
    grsc=30,
    angle=0,
    offs=np.array([0, 0])
)
ax_pwl_traj.pcolor(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)


###############################################################################
# random walk                                                                 #
###############################################################################
print("Random walk")
ox, oy = np.meshgrid(
    np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False),
    np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False)
)
oxr = ox.reshape(1, -1)[0]
oyr = oy.reshape(1, -1)[0]
oxr, oyr = convert_to_rhombus(oxr, oyr)
# random walk firing rate subplot
ax_rw = fig.add_subplot(spec[4:6, 3:5])
ax_rw.spines['top'].set_visible(False)
ax_rw.spines['right'].set_visible(False)
mfrrw_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "mfrrw.pkl"
)
trajec = traj(settings.dt, settings.tmax, sp=settings.speed)
if not os.path.isfile(mfrrw_fname):
    direc_binned, meanfr, _, summed_fr = gridpop_repsupp(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    with open(mfrrw_fname, 'wb') as f:
        pickle.dump(meanfr, f)
else:
    with open(mfrrw_fname, 'rb') as f:
        meanfr = pickle.load(f)
plt.plot(np.linspace(0, 360, settings.phbins), meanfr, 'k')
plt.xlabel(r'Movement direction ($^\circ$)', fontsize=settings.fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=settings.fs)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=settings.fs)
plt.yticks(yticksnums, fontsize=settings.fs)
plt.ylim([750, 900])


print("Hex for rw")
# random walk hexasymmetry
grsrw_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "grsrw.pkl"
)
grs_rw = np.array([])
grs_path_rw = np.array([])

i = 0
if os.path.exists(grsrw_fname):
    with open(grsrw_fname, 'rb') as f:
        grs_rw = pickle.load(f)
        if extend:
            while i < imax:
                print(i)
                # ox, oy = gen_offsets(N=settings.N)
                # oxr, oyr = convert_to_rhombus(ox, oy)
                ox, oy = np.meshgrid(
                    np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False),
                    np.linspace(0, 1, int(np.sqrt(settings.N)), endpoint=False)
                )
                oxr = ox.reshape(1, -1)[0]
                oyr = oy.reshape(1, -1)[0]
                oxr, oyr = convert_to_rhombus(oxr, oyr)
                trajec = traj(settings.dt, settings.tmax, sp=settings.speed)
                direc_binned, meanfr, _, summed_fr = gridpop_repsupp(
                    settings.N,
                    grsc=settings.grsc,
                    phbins=settings.phbins,
                    traj=trajec,
                    oxr=oxr,
                    oyr=oyr
                )
                gr60_rw = get_hexsym(summed_fr, trajec)
                gr60_path_rw = get_pathsym(trajec)
                grs_rw = np.hstack((grs_rw, gr60_rw))
                grs_path_rw = np.hstack((grs_path_rw, gr60_path_rw))
                i += 1

            with open(grsrw_fname, 'wb') as f:
                pickle.dump(grs_rw, f)
else:
    while i < imax:
        print(i)
        ox, oy = gen_offsets(N=settings.N)
        oxr, oyr = convert_to_rhombus(ox, oy)
        trajec = traj(settings.dt, settings.tmax, sp=settings.speed)
        direc_binned, meanfr, _, summed_fr = gridpop_repsupp(
            settings.N,
            grsc=settings.grsc,
            phbins=settings.phbins,
            traj=trajec,
            oxr=oxr,
            oyr=oyr
        )
        gr60_rw = get_hexsym(summed_fr, trajec)
        gr60_path_rw = get_pathsym(trajec)
        grs_rw = np.hstack((grs_rw, gr60_rw))
        grs_path_rw = np.hstack((grs_path_rw, gr60_path_rw))
        i += 1

    with open(grsrw_fname, 'wb') as f:
        pickle.dump(grs_rw, f)


grs_rw = np.median(grs_rw)
plt.text(
    0.48,
    0.85,
    'H = '+str(np.round(grs_rw, 1))+" spk/s",
    fontsize=settings.fs,
    transform=ax_rw.transAxes
)


# random walk trajectory subplot
rw_traj_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "rw_traj.pkl"
)
ax_rw_traj = fig.add_subplot(spec[4:6, 5])

# with open(rw_traj_fname, 'rb') as f:
#         trajec = pickle.load(f)

# trajec = traj(settings.dt, settings.tmax2, sp=settings.speed)

if not os.path.isfile(rw_traj_fname):
    trajec = traj(
        settings.dt,
        settings.tmax,
        sp=settings.speed,
        dphi=settings.dphi
    )
    os.makedirs(os.path.dirname(rw_traj_fname), exist_ok=True)
    with open(rw_traj_fname, 'wb') as f:
        pickle.dump(trajec, f)
else:
    with open(rw_traj_fname, 'rb') as f:
        trajec = pickle.load(f)

plt.plot(trajec[1][:part], trajec[2][:part], trajc, linewidth=1.5)
ax_rw_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
plt.axis('square')


xtr, ytr, dirtr = trajec[1][:part], trajec[2][:part], trajec[3][:part]
tol = 5 * np.pi / 180
ax_rw_traj.pcolor(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)


bar_dist = 3 * settings.grsc
bar_ang = 240
bar_len = 4 * settings.grsc
bar_off = -5 * settings.grsc
plt.plot(
    [
        bar_dist * np.cos(2*np.pi*bar_ang/360) + 2 * settings.grsc + bar_off,
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len + 2 * settings.grsc
        + bar_off
    ],
    [
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360)
    ],
    color="red",
    lw=3
)


###############################################################################
# parameter search                                                            #
###############################################################################
taus = settings.taus
ws = settings.ws
nxticks = 4
xtickstep = int(len(taus) / nxticks)
ytickstep = 5


gr60s_fname = os.path.join(
    settings.loc,
    "repsupp",
    "fig5",
    "meanfr",
    "gr60s.pkl"
)
gr60s = pickle.load(open(gr60s_fname, "rb"))


# parameter search
ax_psearch = fig.add_subplot(spec[3:, 0:3])
# ax_psearch.set_aspect('equal')
plt.rcParams.update({'font.size': int(settings.fs)})
pcolor_psearch = plt.pcolor(
    taus,
    ws,
    np.median(gr60s, axis=-1).T,
    cmap="viridis"
)
gr60s = gaussian_filter(np.median(gr60s, axis=-1).T, settings.smooth_sigma*1.2)
CS = plt.contour(taus, ws, gr60s, levels=[2, 6, 10, 14], colors='white')
manual_locations = [
    (6, 0.7), (8, 0.6), (15, 0.40), (27, 0.25)
]
ax_psearch.clabel(CS, inline=True, fontsize=14, manual=manual_locations)
clip = plt.scatter(
    settings.tau_rep,
    settings.w_rep,
    c='red',
    s=150,
    zorder=10,
    marker="o"
)
clip.set_clip_on(False)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=settings.fs)
plt.yticks(fontsize=settings.fs)
plt.xlabel("Adaptation time constant $\\tau_r$ (s)", fontsize=int(settings.fs))
plt.ylabel("Adaptation\nweight $w_r$", fontsize=int(settings.fs))
plt.title("star-like run", y=1.00, pad=14)


###############################################################################
# Variability                                                                 #
###############################################################################
# Ratemap
# ax_ratemap = fig.add_subplot(spec[5, 0])

# # Visualisation of rate map
# ox_rand, oy_rand = gen_offsets(N=1, kappacl=0.)
# oxr_rand, oyr_rand = convert_to_rhombus(ox_rand, oy_rand)
# trajec = traj(
#     settings.dt,
#     settings.tmax / 10,
#     bound=200.,
#     sq_bound=True
# )
# x, y = trajec[1], trajec[2]
# signal = gridpop_repsupp(
#     1,
#     settings.grsc,
#     settings.phbins,
#     trajec,
#     oxr_rand,
#     oyr_rand
# )[-1]

# x_bin_edges = np.arange(min(x), max(x), 3)
# y_bin_edges = np.arange(min(y), max(y), 3)

# avg, xedges, yedges = bin_avg(
#     x, y, signal, x_bin_edges, y_bin_edges
# )

# X, Y = np.meshgrid(x_bin_edges[:-1], y_bin_edges[:-1])
# smooth = smooth_with_gaussian(avg, 1.5)

# ax_ratemap.pcolormesh(X, Y, smooth.T)
# # plt.imshow(smooth)
# ax_ratemap.plot(x, y, color="white", linewidth=1.)
# plt.gca().set_aspect('equal', adjustable='box')

# factor = 0.33
# plt.xlim(min(x)*factor, max(x)*factor)
# plt.ylim(min(y)*factor, max(y)*factor)
# plt.xticks([])
# plt.yticks([])


# # Coefficient of variation
# ax_cov = fig.add_subplot(spec[5, 1:3])

# alldata = np.load(
#     os.path.join(
#         settings.loc,
#         "repsupp",
#         "repsupp_variability",
#         "variability",
#         "peak_data.npy"
#     ),
#     allow_pickle=True
# )

# rep = len(alldata)
# N = len(alldata[0][0])

# null_peaks = [i[0] for i in alldata]
# repsupp_peaks = [i[1] for i in alldata]
# null_cvs = np.zeros((rep, N))
# repsupp_cvs = np.zeros((rep, N))

# for i in range(rep):
#     for j in range(N):
#         # print(null_peaks[i][j])
#         null_cvs[i, j] = np.std(
#             np.array(null_peaks[i][j])
#         )/np.mean(
#             np.array(null_peaks[i][j])
#         )
#         repsupp_cvs[i, j] = np.std(
#             np.array(repsupp_peaks[i][j])
#         )/np.mean(
#             np.array(repsupp_peaks[i][j])
#         )

# bin_edges = np.linspace(0.4, 0.55, 100)

# # Coefficient of variation for all grid cells
# ax_cov.hist(
#     null_cvs.flatten(),
#     bin_edges,
#     label="Null",
#     alpha=0.4
# )
# ax_cov.hist(
#     repsupp_cvs.flatten(),
#     bin_edges,
#     label="Repetition-\nsuppression",
#     alpha=0.4
# )
# plt.xlabel("Coefficient of variation")
# plt.ylabel("# of cells")
# box = ax_cov.get_position()
# ax_cov.set_position([box.x0, box.y0, box.width * 0.7, box.height])

# # Put a legend to the right of the current axis
# ax_cov.legend(
#     loc='center left', bbox_to_anchor=(1, 0.5), fontsize=settings.fs*0.8
# )
# # plt.yticks([])
# # plt.show()


###############################################################################
# figure layout tweaks                                                        #
###############################################################################
plt.subplots_adjust(
    wspace=3.2,
    hspace=3.2
)


# posparam = ax_psearch.get_position()
# pointsparam = posparam.get_points()
# mean_pointsparam = np.array(
#     [
#         (pointsparam[0][0] + pointsparam[1][0])/2,
#         (pointsparam[0][1] + pointsparam[1][1])/2
#     ]
# )
# pointsparam -= mean_pointsparam
# pointsparam[0][0] *= 0.65
# pointsparam[1][0] *= 0.8
# pointsparam[0][1] *= 1.2
# pointsparam[1][1] *= 1.2
# pointsparam += mean_pointsparam
# posparam.set_points(pointsparam)
# ax_psearch.set_position(posparam)

# ax_pos(ax_fr, 0., 0., 0.6, 1.)
# ax_pos(ax_psearch, 0., 0.12, 0.7, 1.1)
# ax_pos(ax_ratemap, 0., 0.02, 4.9, 4.9)
# ax_pos(ax_cov, -0.060, 0.03, 0.80, 4)


div_psearch = make_axes_locatable(ax_psearch)
cax_psearch = div_psearch.append_axes('right', size='2%', pad=0.05)
cbar_psearch = fig.colorbar(
    pcolor_psearch,
    cax=cax_psearch,
    fraction=0.020,
    pad=0.04,
    ticks=[2, 6, 10, 14]
)
cbar_psearch.set_label("Hexasymmetry\n(spk/s)", fontsize=int(settings.fs))

ax_pos(ax_psearch, 0., 0.12, 0.825, 0.66)
ax_pos(ax_prefdir, 0., 0., 1.6, 1.6)


div_prefdir = make_axes_locatable(ax_prefdir)
cax_prefdir = div_prefdir.append_axes('right', size='8%', pad=0.05)
cbar_prefdir = fig.colorbar(
    pcolor_prefdir, cax=cax_prefdir, fraction=0.020, pad=0.04
)
cbar_prefdir.set_ticks([0, 8])
cbar_prefdir.set_label('Firing rate\n(spk/s)', fontsize=settings.fs)
cbar_prefdir.ax.tick_params(labelsize=settings.fs)


ax_pos(ax_star_traj, -0.05, 0., 4.5, 4.5)
ax_pos(ax_pwl_traj, -0.05, -0.02, 4.5, 4.5)
ax_pos(ax_rw_traj, -0.05, -0.04, 4.5, 4.5)

# ax_pos(ax_star, 0., 0., 1.2, 0.5)
# ax_pos(ax_pw, 0., -0.02, 1.2, 0.5)
# ax_pos(ax_rw, 0., -0.04, 1.2, 0.5)

ax_pos(ax_star, 0., 0., 1., 1.)
ax_pos(ax_pw, 0., -0.02, 1., 1.)
ax_pos(ax_rw, 0., -0.04, 1., 1.)

# plt.tight_layout


plt.savefig(
    os.path.join(
        settings.loc,
        "repsupp",
        "fig5",
        'Figure4_repsupp.png'
    ),
    dpi=300
)
plt.close()
