import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import settings
import os
import pickle
from utils.grid_funcs import (
    traj,
    traj_pwl,
    gridpop_clustering,
    gen_offsets
)
from utils.utils import (
    grid_meanfr,
    get_hexsym,
    get_pathsym,
    convert_to_rhombus,
    ax_pos
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')


kappacl = 10.
N = 1024
figsize = (20, 10)
fs = settings.fs
meanoff = settings.meanoff
imax = 10


ox, oy = gen_offsets(N=N, kappacl=kappacl)
oxr, oyr = convert_to_rhombus(ox, oy)


rmax = settings.rmax
bins = settings.bins
phbins = settings.phbins
amax = settings.amax


part = 4000


trajc = "w"
pointc = "r"


plot_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "figure4_clustering.png"
)
starsym_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "starsym.pkl"
)
pwsym_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "pwsym.pkl"
)
rwsym_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "rwsym.pkl"
)


pwl_traj_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "pwl_traj.pkl"
)
rw_traj_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "rw_traj.pkl"
)


cluststar_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "conjunctive_sw_N1024_nphi360_rmin0_rmax3.npy"
        )
)
clustpwl_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "clustering_pwl_N1024_nphi360_rmin0_rmax3_kappa10.npy"
        )
)
clustrw_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            (
                "clustering_rw_N1024_nphi360_T9000_dt0_01_v0_1_sigmatheta0_5"
                "_kappa10.npy"
            )
        )
)
with open(cluststar_afname, 'rb') as f:
    amfr_cluststar = np.load(f)
with open(clustpwl_afname, 'rb') as f:
    amfr_clustpwl = np.load(f)
with open(clustrw_afname, 'rb') as f:
    amfr_clustrw = np.load(f)

plt.figure(figsize=figsize)

# star-like run trajectory
ax_star_traj = plt.subplot(5, 5, 1)


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
    trajc
)
ax_star_traj.set_aspect('equal', adjustable='box')
plt.xticks([0])
plt.yticks([0])
X_bgr, Y_bgr, _ = np.meshgrid(
    np.linspace(-120, 120, 1000),
    np.linspace(-120, +120, 1000),
    1
)
gr_bgr = grid_meanfr(X_bgr, Y_bgr, grsc=30, angle=0, offs=np.array([0, 0]))
ax_star_traj.yaxis.set_ticks_position("right")
ax_star_traj.pcolor(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)

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

plt.axis("square")


# star-like run firing rate
ax_star_fr = plt.subplot(5, 5, (2, 3))

star_fr_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "star_fr.pkl"
)

if not os.path.exists(star_fr_fname):
    r, phi, indoff = np.meshgrid(
        np.linspace(0, rmax, bins),
        np.linspace(0, 2 * np.pi, phbins, endpoint=False),
        np.arange(len(oxr))
    )
    X, Y = r * np.cos(phi), r*np.sin(phi)
    grids = amax * grid_meanfr(X, Y, offs=(oxr, oyr))
    gr2 = np.sum(grids, axis=2)
    meanfr1 = np.sum(np.sum(grids, axis=1), axis=1) / bins
    os.makedirs(os.path.dirname(star_fr_fname), exist_ok=True)
    with open(star_fr_fname, "wb") as f:
        pickle.dump((meanfr1), f)
else:
    with open(star_fr_fname, "rb") as f:
        meanfr1 = pickle.load(f)

plt.plot(np.linspace(0, 360, phbins), meanfr1, 'k', zorder=10)
plt.plot(
    np.linspace(0, 360, settings.phbins),
    amfr_cluststar, color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
# plt.xlabel(r'Movement direction ($^\circ$)', fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [], fontsize=fs)
# plt.xticks([])
plt.yticks(fontsize=fs)
ymax = 1.5*max(meanfr1)
plt.ylim([0, ymax])
# plt.text(0.2,0.8,'kappa = 10',fontsize=fs,transform=ax_star_fr.transAxes)
ax_star_fr.spines['top'].set_visible(False)
ax_star_fr.spines['right'].set_visible(False)
# ax_star_fr.spines['bottom'].set_visible(False)
# ax_star_fr.spines['left'].set_visible(False)


grs_star = np.array([])
grs_path_star = np.array([])

print("Doing hex for star-like run")
if not os.path.exists(starsym_fname):
    i = 0
    while i < imax:
        print(i)
        ox, oy = np.random.vonmises(
            2*np.pi*(meanoff[0]-0.5), kappacl, int(N)
        ) / 2. / np.pi + 0.5, np.random.vonmises(
            2*np.pi*(meanoff[1]-0.5), kappacl, int(N)
        ) / 2. / np.pi + 0.5
        oxr, oyr = convert_to_rhombus(ox, oy)
        grids = amax * grid_meanfr(X, Y, offs=(oxr, oyr))
        gr2 = np.sum(grids, axis=2)
        meanfr1 = np.sum(np.sum(grids, axis=1), axis=1) / bins
        gr60_star = np.abs(np.sum(gr2 * np.exp(-6j * phi[:, :, 0]))) \
            / np.size(gr2)
        gr60_path_star = np.abs(np.sum(np.exp(-6j * phi[:, :, 0]))) \
            / np.size(gr2)
        grs_star = np.hstack((grs_star, gr60_star))
        grs_path_star = np.hstack((grs_path_star, gr60_path_star))
        i += 1
    os.makedirs(os.path.dirname(starsym_fname), exist_ok=True)
    with open(starsym_fname, 'wb') as f:
        pickle.dump([grs_star, grs_path_star], f)
else:
    with open(starsym_fname, 'rb') as f:
        grs_star, grs_path_star = pickle.load(f)


plt.text(
    0.6,
    0.75,
    'H = '+str(np.round(np.mean(grs_star), 1))+" spk/s",
    fontsize=fs, transform=ax_star_fr.transAxes
)


ax_offs = inset_axes(
    ax_star_fr,
    width='15%',
    height='40%',
    loc=2,
    borderpad=1.5
)
plt.scatter(ox+0.5*oy, np.sqrt(3)/2*oy, s=5, c='k')
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
plt.xlim([-0.05, 1.5])
plt.ylim([-0.05, 1])
plt.xticks([])
plt.yticks([])
# plt.xlabel('Grid phases',fontsize=fs)
ax_offs.axis("off")
# plt.text(0.1,-0.2,'kappa = 10',fontsize=fs)
ax_offs.set_aspect('equal')


# pwl trajectory
if not os.path.isfile(pwl_traj_fname):
    trajec = traj_pwl(
        settings.phbins,
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


ax_pwl_traj = plt.subplot(5, 5, 6)
plt.plot(
    trajec[1][:int(part*2)], trajec[2][:int(part*2)], trajc, linewidth=1.5
)
ax_pwl_traj.set_aspect('equal', adjustable='box')
plt.axis('square')
plt.xticks([])
plt.yticks([])
X_bgr, Y_bgr, _ = np.meshgrid(
    np.linspace(-2000, 2000, 4000),
    np.linspace(-2000, 2000, 4000),
    1
)
gr_bgr = grid_meanfr(X_bgr, Y_bgr, grsc=30, angle=0, offs=np.array([0, 0]))
ax_pwl_traj.pcolor(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)

bar_dist = 4 * settings.grsc
bar_ang = -120
plt.plot(
    [
        bar_dist * np.cos(2*np.pi*bar_ang/360),
        bar_dist * np.cos(2*np.pi*bar_ang/360) - bar_len,
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len
    ],
    [
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360)
    ],
    color="red",
    lw=3
)


# pwl firing rate
ax_pwl_fr = plt.subplot(5, 5, (7, 8))


pwl_fr_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "pwl_fr.pkl"
)


if not os.path.isfile(pwl_fr_fname):
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    os.makedirs(os.path.dirname(pwl_fr_fname), exist_ok=True)
    with open(pwl_fr_fname, 'wb') as f:
        pickle.dump(fr_mean, f)
else:
    with open(pwl_fr_fname, 'rb') as f:
        fr_mean = pickle.load(f)


plt.plot(
    np.linspace(0, 360, settings.phbins, endpoint=False),
    fr_mean,
    'k',
    zorder=10
)
plt.plot(
    np.linspace(0, 360, settings.phbins, endpoint=False),
    amfr_clustpwl,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
plt.xticks([0, 60, 120, 180, 240, 300, 360], [], fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylim([0, ymax])
ax_pwl_fr.spines['top'].set_visible(False)
ax_pwl_fr.spines['right'].set_visible(False)


grs_pl = np.array([])
grs_path_pl = np.array([])
print("Doing hex for pwl run")
if not os.path.exists(pwsym_fname):
    i = 0
    while i < imax:
        print(i)
        trajec = traj_pwl(
            settings.phbins,
            settings.rmax,
            settings.dt,
            sp=settings.speed
        )
        direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
            N=settings.N,
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
    os.makedirs(os.path.dirname(pwsym_fname), exist_ok=True)
    with open(pwsym_fname, 'wb') as f:
        pickle.dump([grs_pl, grs_path_pl], f)
else:
    with open(pwsym_fname, 'rb') as f:
        grs_pl, grs_path_pl = pickle.load(f)


plt.text(
    0.6,
    0.7,
    'H = '+str(np.round(np.median(grs_pl), 1))+" spk/s",
    fontsize=fs,
    transform=ax_pwl_fr.transAxes
)

ax_pwloffs = inset_axes(
    ax_pwl_fr,
    width='15%',
    height='40%',
    loc=2,
    borderpad=1.5
)
plt.scatter(ox+0.5*oy, np.sqrt(3)/2*oy, s=5, c='k')
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
plt.xlim([-0.05, 1.5])
plt.ylim([-0.05, 1])
plt.xticks([])
plt.yticks([])
ax_pwloffs.axis("off")
ax_pwloffs.set_aspect('equal')


# random walks
# random walk traj
if not os.path.isfile(rw_traj_fname):
    trajec = traj(
        dt=settings.dt,
        tmax=settings.tmax,
        sp=settings.speed
    )
    os.makedirs(os.path.dirname(rw_traj_fname), exist_ok=True)
    with open(rw_traj_fname, 'wb') as f:
        pickle.dump(trajec, f)
else:
    with open(rw_traj_fname, 'rb') as f:
        trajec = pickle.load(f)


ax_rw_traj = plt.subplot(5, 5, 11)
plt.plot(trajec[1][:part], trajec[2][:part], trajc, linewidth=1.5)
# plt.plot(trajec[1],trajec[2],'k')
ax_rw_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
plt.axis('square')
# xtr,ytr,dirtr = trajec[1][:part], trajec[2][:part], trajec[3][:part]
# tol = 5 * np.pi / 180
# ax_rw_traj.scatter(
#     xtr[abs(dirtr)<tol],
#     ytr[abs(dirtr)<tol],
#     c=pointc,
#     marker='.',
#     s=50,
#     zorder=5
# )
# plt.xticks([0])
# plt.yticks([0])
ax_rw_traj.pcolor(
    X_bgr[:, :, 0], Y_bgr[:, :, 0], gr_bgr[:, :, 0], shading='auto'
)
bar_dist = 2 * settings.grsc
bar_ang = 60
bar_len = 4 * settings.grsc
plt.plot(
    [
        bar_dist * np.cos(2*np.pi*bar_ang/360),
        bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len
    ],
    [
        bar_dist * np.sin(2*np.pi*bar_ang/360),
        bar_dist * np.sin(2*np.pi*bar_ang/360)
    ],
    color="red",
    lw=3
)


# random walk firing rate
ax_rw_fr = plt.subplot(5, 5, (12, 13))

rw_fr_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "rw_fr.pkl"
)


if not os.path.isfile(rw_fr_fname):
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    os.makedirs(os.path.dirname(rw_fr_fname), exist_ok=True)
    with open(rw_fr_fname, 'wb') as f:
        pickle.dump(fr_mean, f)
else:
    with open(rw_fr_fname, 'rb') as f:
        fr_mean = pickle.load(f)


plt.plot(
    np.linspace(0, 360, settings.phbins, endpoint=False),
    fr_mean,
    'k',
    zorder=10
)
plt.plot(
    np.linspace(0, 360, settings.phbins, endpoint=False),
    amfr_clustrw,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
plt.xlabel(r'Movement direction ($^\circ$)', fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
plt.xticks([0, 60, 120, 180, 240, 300, 360], fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylim([0, ymax])
ax_rw_fr.spines['top'].set_visible(False)
ax_rw_fr.spines['right'].set_visible(False)


grs_rw = np.array([])
grs_path_rw = np.array([])
print("Doing hex for rw run")
if not os.path.exists(rwsym_fname):
    i = 0
    while i < imax:
        print(i)
        trajec = traj(
            dt=settings.dt,
            tmax=settings.tmax,
            sp=settings.speed
        )
        direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
            N=settings.N,
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
    os.makedirs(os.path.dirname(rwsym_fname), exist_ok=True)
    with open(rwsym_fname, 'wb') as f:
        pickle.dump([grs_rw, grs_path_rw], f)
else:
    with open(rwsym_fname, 'rb') as f:
        grs_rw, grs_path_rw = pickle.load(f)


plt.text(
    0.6,
    0.6,
    'H = '+str(np.round(np.median(grs_rw), 1))+" spk/s",
    fontsize=fs,
    transform=ax_rw_fr.transAxes
)


ax_rwoffs = inset_axes(
    ax_rw_fr,
    width='15%',
    height='40%',
    loc=2,
    borderpad=1.5
)
plt.scatter(ox+0.5*oy, np.sqrt(3)/2*oy, s=5, c='k')
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
plt.xlim([-0.05, 1.5])
plt.ylim([-0.05, 1])
plt.xticks([])
plt.yticks([])
ax_rwoffs.axis("off")
ax_rwoffs.set_aspect('equal')


###############################################################################
# phase dependence stuff                                                      #
###############################################################################
res = 101
N_phasedep = 1
x = np.linspace(0, 1, res, endpoint=True)
y = np.linspace(0, 1, res, endpoint=True)
Xr, Yr = np.meshgrid(x, y)
Xc, Yc = convert_to_rhombus(Xr, Yr)
gr60s_starphd = np.zeros((len(x), len(y)))
ph60s_starphd = np.zeros((len(x), len(y)))


r, phi, indoff = np.meshgrid(
    np.linspace(0, settings.rmax, settings.bins),
    np.linspace(0, 2 * np.pi, phbins, endpoint=False),
    np.arange(N_phasedep)
)
X, Y = r * np.cos(phi), r*np.sin(phi)


star_phd_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "phasedep_star.pkl"
)


if not os.path.isfile(star_phd_fname):
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            print("star phd: ", ix, iy)
            oxr_1, oyr_1 = convert_to_rhombus(np.array([xx]), np.array([yy]))
            # ox_1, oy_1 = gen_offsets(N_phasedep, settings.kappa_si, (xx, yy))
            # oxr_1, oyr_1 = convert_to_rhombus(ox_1, oy_1)
            grids = settings.amax * grid_meanfr(X, Y, offs=(oxr_1, oyr_1))
            gr2 = np.sum(grids, axis=2)
            meanfr1 = np.sum(np.sum(grids, axis=1), axis=1) / bins

            gr60s_starphd[ix, iy] = np.abs(
                np.sum(gr2 * np.exp(-6j * phi[:, :, 0]))
            ) / np.size(gr2)

            fr2 = np.tile(meanfr1, 3)
            ff = np.linspace(0, 1./2., 3*settings.phbins//2)
            ph = np.angle(np.fft.fft(fr2))[:len(fr2)//2]
            ph60s_starphd[ix, iy] = ph[np.argmin(abs(ff-1./60*360/phbins))] \
                / 2 / np.pi*60
    os.makedirs(os.path.dirname(star_phd_fname), exist_ok=True)
    with open(star_phd_fname, 'wb') as f:
        pickle.dump([gr60s_starphd, ph60s_starphd], f)
else:
    with open(star_phd_fname, 'rb') as f:
        gr60s_starphd, ph60s_starphd = pickle.load(f)


# if not os.path.exists(star_fr_fname):
#     r,phi,indoff = np.meshgrid(
#         np.linspace(0,rmax,bins),
#         np.linspace(0,2 * np.pi,phbins, endpoint=False),
#         np.arange(len(oxr))
#     )
#     X,Y = r * np.cos(phi), r*np.sin(phi)
#     grids = amax * grid_meanfr(X, Y, offs=(oxr,oyr))
#     gr2 = np.sum(grids, axis=2)
#     meanfr1 = np.sum(np.sum(grids,axis=1),axis=1) / bins
#     os.makedirs(os.path.dirname(star_fr_fname), exist_ok=True)
#     with open(star_fr_fname, "wb") as f:
#         pickle.dump((meanfr1), f)
# else:
#     with open(star_fr_fname, "rb") as f:
#         meanfr1 = pickle.load(f)


# phase dependence of starlike walks ##########################################
# gr60s = np.array([])
# starphdep_fname = os.path.join(
#     settings.loc,
#     "clustering",
#     "fig4",
#     "phasedependence_gr60_star.txt"
# )
# starphhex_fname = os.path.join(
#     settings.loc,
#     "clustering",
#     "fig4",
#     "phasedependence_ph60_star.txt"
# )
# f = open(starphdep_fname, "r")
# for i,x in enumerate(f):
#     gr60s = np.append(gr60s, np.array([float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']))
# f.close()
# gr60s = gr60s.reshape((res,res))

# ph60s = np.array([])
# f = open(starphhex_fname, "r")
# for i,x in enumerate(f):
#     ph60s = np.append(ph60s, np.array([float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']))
# f.close()
# ph60s = ph60s.reshape((res,res))


# phase dependency of hexasymmetry
ax_star_phdep = plt.subplot(5, 5, 4)
ax = plt.gca()
# im = ax.pcolormesh(Xc,Yc,N*gr60s_starphd,vmin=0, vmax=60, shading='auto')
im = ax.pcolormesh(
    Xc, Yc, N * gr60s_starphd / N_phasedep, vmin=0, vmax=60, shading='auto'
)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax.set_aspect('equal')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax)
cb.set_label('Hexasymmetry\n(spk/s)', fontsize=0.9*fs)


# phase of hexasymmetry
ax_star_phhex = plt.subplot(5, 5, 5)
ax = plt.gca()
im = ax.pcolormesh(
    Xc,
    Yc,
    ph60s_starphd / np.pi * 30,
    cmap='twilight_shifted',
    vmin=-30,
    vmax=30,
    shading='auto'
)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax.set_aspect('equal')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, ticks=[-30, 30])
cb.set_label('Apparent\npreferred grid\norientation\n$(^\\circ)$', fontsize=fs)


# phase dependence of piece-wise linear walks #################################
pwl_phasedep_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "phasedep_pwl.pkl"
)


reso = 101
x = np.linspace(0, 1, reso, endpoint=True)
y = np.linspace(0, 1, reso, endpoint=True)
X, Y = np.meshgrid(x, y)
Xc, Yc = convert_to_rhombus(X, Y)
gr60s = np.zeros((len(x), len(y)))
ph60s = np.zeros((len(x), len(y)))


if not os.path.isfile(pwl_phasedep_fname):
    for ix, xx in enumerate(x):
        print(ix)
        for iy, yy in enumerate(y):
            oxr_1, oyr_1 = convert_to_rhombus(np.array([xx]), np.array([yy]))
            trajec = traj_pwl(
                settings.phbins,
                settings.rmax,
                settings.dt,
                sp=settings.speed
            )
            direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
                N=1,
                grsc=settings.grsc,
                phbins=settings.phbins,
                traj=trajec,
                oxr=oxr_1,
                oyr=oyr_1
            )
            fr2 = np.tile(fr_mean, 3)
            ff = np.linspace(0, 1./2., 3*settings.phbins//2)
            ph = np.angle(np.fft.fft(fr2))[:len(fr2)//2]
            p60 = ph[np.argmin(abs(ff-1./60*360/phbins))]/2/np.pi*60

            gr60s[ix, iy] = get_hexsym(summed_fr, trajec)
            ph60s[ix, iy] = p60
    os.makedirs(os.path.dirname(pwl_phasedep_fname), exist_ok=True)
    with open(pwl_phasedep_fname, 'wb') as f:
        pickle.dump([gr60s, ph60s], f)
else:
    with open(pwl_phasedep_fname, 'rb') as f:
        gr60s, ph60s = pickle.load(f)


# phase dependency of hexasymmetry
ax_pwl_phdep = plt.subplot(5, 5, 9)
ax = plt.gca()
im = ax.pcolormesh(Xc, Yc, N*gr60s, vmin=0, vmax=60, shading='auto')
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax.set_aspect('equal')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax)
cb.set_label('Hexasymmetry\n(spk/s)', fontsize=0.9*fs)


# phase of hexasymmetry
ax_pwl_phhex = plt.subplot(5, 5, 10)
ax = plt.gca()
im = ax.pcolormesh(
    Xc,
    Yc,
    ph60s / np.pi * 30,
    cmap='twilight_shifted',
    vmin=-30,
    vmax=30,
    shading='auto'
)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax.set_aspect('equal')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, ticks=[-30, 30])
cb.set_label('Apparent\npreferred grid\norientation\n$(^\\circ)$', fontsize=fs)


# phase dependence of random walks ############################################
res = 101
x = np.linspace(0, 1, res, endpoint=True)
y = np.linspace(0, 1, res, endpoint=True)
X, Y = np.meshgrid(x, y)
Xc, Yc = convert_to_rhombus(X, Y)
rwphdep_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "phasedependence_gr60_rw.txt"
)
rwphhex_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "phasedependence_ph60_rw.txt"
)
gr60s = np.array([])
f = open(rwphdep_fname, "r")
for i, x in enumerate(f):
    gr60s = np.append(
        gr60s,
        np.array(
            [float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']
        )
    )
f.close()
gr60s = gr60s.reshape((res, res))

ph60s = np.array([])
f = open(rwphhex_fname, "r")
for i, x in enumerate(f):
    ph60s = np.append(
        ph60s,
        np.array(
            [float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']
        )
    )
f.close()
ph60s = ph60s.reshape((res, res))


# phase dependency of hexasymmetry
ax_rw_phdep = plt.subplot(5, 5, 14)
ax = plt.gca()
im = ax.pcolormesh(Xc, Yc, N*gr60s, vmin=0, vmax=60, shading='auto')
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax.set_aspect('equal')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax)
cb.set_label('Hexasymmetry\n(spk/s)', fontsize=0.9*fs)


# phase of hexasymmetry
ax_rw_phhex = plt.subplot(5, 5, 15)
ax = plt.gca()
im = ax.pcolormesh(
    Xc,
    Yc,
    ph60s / np.pi * 30,
    cmap='twilight_shifted',
    vmin=-30,
    vmax=30,
    shading='auto'
)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax.set_aspect('equal')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, ticks=[-30, 30])
cb.set_label('Apparent\npreferred grid\norientation\n$(^\circ)$', fontsize=fs)


###############################################################################
# Random field simulations                                                    #
###############################################################################
"""
ph1_gauss = np.array([])
ph2_gauss = np.array([])
ph1_grid = np.array([])
ph2_grid = np.array([])


ph1_gauss_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "ph1c_gauss.txt"
)
ph2_gauss_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "ph2c_gauss.txt"
)
ph1_grid_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "ph1_grid.txt"
)
ph2_grid_fname = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "ph2_grid.txt"
)


f = open(ph1_gauss_fname, "r")
for i,x in enumerate(f):
    ph1_gauss = np.append(
        ph1_gauss,
        np.array(
            [float(elem) for elem in x.replace(
                '[',''
                ).replace(
                    ']',''
                ).replace(
                    '\n',''
                ).split(
                    ' '
                ) if elem!='']
            )
        )
f.close()
res_rf = int(np.sqrt(len(ph1_gauss)))
ph1_gauss = ph1_gauss.reshape((res_rf,res_rf))


f = open(ph2_gauss_fname, "r")
for i,x in enumerate(f):
    ph2_gauss = np.append(ph2_gauss, np.array([float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']))
f.close()
ph2_gauss = ph2_gauss.reshape((res_rf,res_rf))


f = open(ph1_grid_fname, "r")
for i,x in enumerate(f):
    ph1_grid = np.append(ph1_grid, np.array([float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']))
f.close()
ph1_grid = ph1_grid.reshape((res_rf,res_rf))


f = open(ph2_grid_fname, "r")
for i,x in enumerate(f):
    ph2_grid = np.append(ph2_grid, np.array([float(elem) for elem in x.replace('[','').replace(']','').replace('\n','').split(' ') if elem!='']))
f.close()
ph2_grid = ph2_grid.reshape((res_rf,res_rf))


x = np.linspace(0,3,res_rf,endpoint=True)
y = np.linspace(0,3,res_rf,endpoint=True)
X,Y = np.meshgrid(x,y)


# gaussian random field
ax_gaussfield = plt.subplot(5, 5, 16)
plt.pcolormesh(X, Y, ph2_gauss)
ax_gaussfield.set_aspect('equal')


ax_gridfield = plt.subplot(5, 5, 21)
plt.pcolormesh(X, Y, ph2_grid)
ax_gridfield.set_aspect('equal')
"""


# plt.subplots_adjust(
#     wspace=1.25, 
#     hspace=1.25
# )

# ax_pos(ax_star_traj,)
# ax_pos(ax_pwl_traj,)
# ax_pos(ax_rw_traj,)


# save plot
os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.tight_layout()
plt.savefig(plot_fname, dpi=300)
plt.close()
