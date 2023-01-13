import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions.gridfcts import (
    traj,
    traj_pwl,
    pwliner,
    gridpop_conj,
    gen_offsets
)
from matplotlib.ticker import ScalarFormatter
from utils.utils import (
    grid,
    grid_meanfr,
    grid_2d,
    get_hexsym,
    get_pathsym,
    convert_to_rhombus,
    ax_pos
)
import utils.settings as settings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pickle
from scipy import special

plt.box(False)
figsize = (22, 10)
fs = settings.fs
meanoff = settings.meanoff
imax = 10


ox, oy = gen_offsets(N=settings.N)
oxr, oyr = convert_to_rhombus(ox, oy)


bins = settings.bins
phbins = settings.phbins
amax = settings.amax


tmax_pwl = 72000
dt_pwl = 9
dt_pwl2 = 0.1
dphi_pwl = 2.
part = 4000
phis = np.linspace(0, 2 * np.pi, settings.phbins, endpoint=False)
tmax_star = 30


tmax_rw = 600


hextextloc = (0.6, 1.1)


plot_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    f"figure3_conjunctive.png"
)


fig = plt.figure(figsize=(20,12))
spec = fig.add_gridspec(
    ncols=3, nrows=6, width_ratios=[1,1.5,1.5], height_ratios=[1,1,1,1,1,1]
) 


###############################################################################
# Preferred direction and parameter visualisation                             #
###############################################################################
# preferred direction
ax = fig.add_subplot(spec[0:3,0])
X,Y = np.meshgrid(np.linspace(-50,50,1000),np.linspace(-50,50,1000))
gr = amax*grid_2d(X,Y,grsc=30, angle=0, offs=np.array([0,0]))
plt.pcolor(X,Y,gr,vmin=0,vmax=8, shading='auto')
plt.arrow(
    0,
    0,
    15,
    np.sqrt(3)/2*30,
    color='r',
    length_includes_head=True,
    head_width = 5
)
ax.set_aspect('equal')
plt.xlabel('x',fontsize=fs)
plt.ylabel('y',fontsize=fs)
plt.xticks([])
plt.yticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
c = plt.colorbar(cax=cax, ticks=[0,4,8])
c.set_label('Firing rate (spk/s)',fontsize=fs)
c.ax.tick_params(labelsize=fs)


# params visualisation
ax2 = fig.add_subplot(spec[0:3,1], projection='polar')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=0.8*fs)
ax2.tick_params(axis='both', which='minor', labelsize=0.8*fs)
mu = 30
kappa = 50
direc = np.linspace(0,2*np.pi,360)
r = np.exp(kappa*np.cos(direc-np.pi/180.*mu))/(2*np.pi*special.i0(kappa))
plt.plot(direc,r/np.mean(r)*np.mean(gr),'k',lw=2)
plt.arrow(
    np.pi/3,
    0,
    0,
    15,
    color='r',
    length_includes_head=True,
    head_width=0.1,
    head_length=1,
    zorder = 5
)
ax2.set_rticks([5, 10, 15, 20])
ax2.set_rlabel_position(-22.5)
ax2.set_xticks(np.pi/180.*np.array([0,90,180,270]))
ax2.grid(True)
plt.text(
    1,
    0.8,
    r'$\frac{1}{\sqrt{\kappa_{c}}}$',
    fontsize=settings.fs*1.2,
    transform=ax2.transAxes
)
plt.text(
    1,
    1.05,
    r'$\sigma_c$',
    fontsize=settings.fs,
    transform=ax2.transAxes
)


###############################################################################
# Parameter search                                                            #
###############################################################################
# conjunctive parameter search
print("Doing parameter search...")
def conj_paramsearch(nn):
    kappamax = 50
    widthmax = 20
    sigmamax = 10
    widths = np.linspace(1e-2, widthmax, nn)
    kappas = (180 / np.pi / widths)**2
    stds = np.linspace(0,sigmamax,nn)
    gr60s = np.zeros((nn,nn))
    for i, width in enumerate(widths):
        print(i)
        for j, std in enumerate(stds):
            ox, oy = gen_offsets(N=settings.N, kappacl=0.)
            oxr, oyr = convert_to_rhombus(ox, oy)
            trajec = traj(
                settings.dt,
                tmax_rw,
                sp = settings.speed,
                dphi = settings.dphi
            )
            direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
                settings.N,
                settings.grsc,
                settings.phbins,
                trajec,
                oxr,
                oyr,
                propconj=settings.propconj_i,
                kappa = (180/ np.pi / width)**2,
                jitter=std
            )
            gr60s[i, j] = get_hexsym(summed_fr, trajec)
    return gr60s, kappas, stds


psearch_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "psearch.pkl"
)


ax9 = fig.add_subplot(spec[0:3,2], adjustable='box', aspect="auto")
nn = 81
if not os.path.isfile(psearch_fname):
    gr60s, kappas, stds = conj_paramsearch(nn)
    os.makedirs(os.path.dirname(psearch_fname), exist_ok=True)
    with open(psearch_fname, 'wb') as f:
        pickle.dump([gr60s, kappas, stds], f)
else:
    with open(psearch_fname, 'rb') as f:
        gr60s, kappas, stds = pickle.load(f)
K,S = np.meshgrid(np.sqrt(1/kappas)*180/np.pi,stds)
print(stds)
gr60s = np.nan_to_num(gr60s)
pcolor_psearch = plt.imshow(gr60s.T, extent=(0, 20, 0, 10), origin="lower", aspect=1)
# pcolor_psearch = plt.pcolormesh(K, S, gr60s.T, cmap="viridis", shading='auto')
plt.xticks(fontsize=fs)
plt.yticks([0, 5, 10], fontsize=fs)
plt.xlabel(r'Tuning width $1 / \sqrt{\kappa_c}$ ($^\circ$)',fontsize=fs)
plt.ylabel(r'Jitter $\sigma_c$ ($^\circ$)',fontsize=fs)
plt.scatter(np.sqrt(1/10.)*180/np.pi,3.,c='red', s=150, marker="*")
plt.scatter(np.sqrt(1/25.)*180/np.pi,1.5,c='red', s=150, marker="^")
clip = plt.scatter(
    np.sqrt(1/50.)*180/np.pi,0,c='red',
    s=150,
    marker="o",
    zorder=10
)
clip.set_clip_on(False)
plt.axis([np.sqrt(1/max(kappas))*180/np.pi,20,0,10])
plt.xlim(5, 20)


###############################################################################
# Vary conjunctive cell paremeters                                            #
###############################################################################
trajec = traj(
    settings.dt,
    settings.tmax2,
    sp = settings.speed,
    dphi = settings.dphi
)
ox, oy = gen_offsets(N=settings.N, kappacl=0.)
oxr, oyr = convert_to_rhombus(ox, oy)


# kappa = 50
print("Comparing params...")
ax3 = fig.add_subplot(spec[3,2])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
mfrk50j0_nfname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata1.pkl"
)
mfrk50j0_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma0_kappa50_ratio1.npy"
        )
)
if not os.path.isfile(mfrk50j0_nfname):
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(mfrk50j0_nfname), exist_ok=True)
    with open(mfrk50j0_nfname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(mfrk50j0_nfname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)


with open(mfrk50j0_afname, 'rb') as f:
        amfr_conjk50j0 = np.load(f)
plt.plot(np.linspace(0,360,settings.phbins), fr_mean, 'k', zorder = 10)
plt.plot(
    np.linspace(0,360,settings.phbins),
    amfr_conjk50j0,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
clip = plt.scatter(380, 2500,c='red', s=100, marker="o")
clip.set_clip_on(False)
plt.xticks([0,60,120,180,240,300,360],[])
plt.yticks(fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60,1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax3.transAxes
)
plt.ylim([0,5100])


# kappa = 25
ax4 = fig.add_subplot(spec[4,2])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
frdata2_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata2.pkl"
)
conjk25j5_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "2G_conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma1_5_kappa25_ratio1.npy"
        )
)
if not os.path.isfile(frdata2_fname):
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=25,
        jitter=1.5
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(frdata2_fname), exist_ok=True)
    with open(frdata2_fname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(frdata2_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjk25j5_afname, 'rb') as f:
        amfr_conjk25j5 = np.load(f)
plt.plot(np.linspace(0,360,settings.phbins), fr_mean, 'k', zorder = 10)
plt.plot(
    np.linspace(0,360,settings.phbins),
    amfr_conjk25j5,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
clip = plt.scatter(380, 3000,c='red', s=100, marker="^")
clip.set_clip_on(False)
plt.xticks([0,60,120,180,240,300,360],[])
plt.yticks(fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60,1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax4.transAxes
)
plt.ylim([0,5100])


# kappa = 10
ax5 = fig.add_subplot(spec[5,2])
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
frdata3_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "frdata3.pkl"
)
conjk2p5j10_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "2H_conjunctive_sw_N1024_nphi360_rmin0_rmax3_sigma3_kappa10_ratio1.npy"
        )
)
if not os.path.isfile(frdata3_fname):
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_r,
        jitter=settings.jitterc_r
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(frdata3_fname), exist_ok=True)
    with open(frdata3_fname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(frdata3_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjk2p5j10_afname, 'rb') as f:
        amfr_conjk2p5j10 = np.load(f)
plt.plot(np.linspace(0,360,settings.phbins), fr_mean, 'k', zorder = 10)
plt.plot(
    np.linspace(0,360,settings.phbins),
    amfr_conjk2p5j10,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
clip = plt.scatter(380, 2500,c='red', s=100, marker="*")
clip.set_clip_on(False)
plt.xticks([0,60,120,180,240,300,360],fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel(r'Movement direction ($^\circ$)',fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60,1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax5.transAxes
)
plt.ylim([0,5100])


###############################################################################
# Vary trajectory type                                                        #
###############################################################################
trajc = "w"
pointc = "r"


pwl_traj_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "pwl_traj.pkl"
)
rw_traj_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "rw_traj.pkl"
)


# starlike walk
# star-like run trajectory
ax_star_traj = fig.add_subplot(spec[3,0], adjustable='box', aspect="auto")
# ax_star_traj.spines['top'].set_visible(False)
# ax_star_traj.spines['right'].set_visible(False)
# ax_star_traj.spines['bottom'].set_visible(False)
# ax_star_traj.spines['left'].set_visible(False)
plt.plot([0,0],[-90,90],trajc, linewidth=1.5)
plt.plot([-90,90],[0,0],trajc, linewidth=1.5)
plt.plot(
    [-1/np.sqrt(2) * 90,1/np.sqrt(2) * 90],
    [-1/np.sqrt(2) * 90,1/np.sqrt(2) * 90],
    trajc,
    linewidth=1.5
)
plt.plot(
    [-1/np.sqrt(2) * 90,1/np.sqrt(2) * 90],
    [1/np.sqrt(2) * 90,-1/np.sqrt(2) * 90],
    trajc,
    
)
ax_star_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
X_bgr,Y_bgr, _ = np.meshgrid(
    np.linspace(-120,120,1000),
    np.linspace(-120,+120,1000),
    1
)
gr_bgr = grid_meanfr(X_bgr,Y_bgr,grsc=30, angle=0, offs=np.array([0,0]))
ax_star_traj.yaxis.set_ticks_position("right")
ax_star_traj.pcolor(X_bgr[:, :, 0],Y_bgr[:, :, 0],gr_bgr[:, :, 0], shading='auto')


bar_dist = 4 * settings.grsc
bar_ang = -120
bar_len = 2 * settings.grsc
bar_off = settings.grsc
plt.plot(
    [bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_off, bar_dist * np.cos(2*np.pi*bar_ang/360) - bar_len + bar_off, bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len + bar_off], 
    [bar_dist * np.sin(2*np.pi*bar_ang/360), bar_dist * np.sin(2*np.pi*bar_ang/360), bar_dist * np.sin(2*np.pi*bar_ang/360)],
    color="red",
    lw=3
)


plt.axis("square")


ax6 = fig.add_subplot(spec[3,1], adjustable='box', aspect="auto")
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
# ax6.spines['bottom'].set_visible(False)
# ax6.spines['left'].set_visible(False)
ax6.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
stardata_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "stardata.pkl"
)
conjstar_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            # "conjunctive_sw_N1024_nphi360_rmin0_rmax3.npy"
            "clustering_sw_N1024_nphi360_rmin0_rmax3.npy"
        )
)


if not os.path.isfile(stardata_fname):
    r,phi,indoff = np.meshgrid(
        np.linspace(0,settings.rmax,bins),
        np.linspace(0,2 * np.pi,phbins, endpoint=False),
        np.arange(len(oxr))
    )
    X,Y = r * np.cos(phi), r*np.sin(phi)
    grids = amax * grid_meanfr(X, Y, offs=(oxr,oyr))
    grids2 = grids.copy()

    Nconj = int(settings.propconj_i*settings.N)
    mu = np.mod(
        np.random.randint(0, 6, Nconj) * 60 + 0*np.random.randn(Nconj), 360
    )
    vonmi = np.exp(
        settings.kappac_i*np.cos(
            np.pi/180*(np.linspace(0, 360, phbins)[:, None]-mu[None, :])
        )
    ) / special.i0(settings.kappac_i)
    # change the first Nconj cells
    grids[:, :, :Nconj] = grids[:, :, :Nconj]*vonmi[:, None, :]


    fr_mean_base = np.sum(np.sum(grids2,axis=1)/bins,axis=1)
    fr_mean = np.sum(np.sum(grids,axis=1)/bins,axis=1)
    fr_mean = fr_mean/np.mean(fr_mean)*np.mean(fr_mean_base)


    gr2 = np.sum(grids, axis=2)
    gr60 = np.abs(np.sum(gr2 * np.exp(-6j*phi[:, :, 0])))/np.size(gr2)
    gr60_path = np.abs(np.sum(np.exp(-6j*phi[:, :, 0])))/np.size(gr2)
    gr0 = np.sum(gr2)/np.size(gr2)

    os.makedirs(os.path.dirname(stardata_fname), exist_ok=True)
    with open(stardata_fname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(stardata_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjstar_afname, 'rb') as f:
        amfr_conjstar = np.load(f)


plt.plot(np.linspace(0,360,settings.phbins), fr_mean, 'k', zorder = 10)
plt.plot(
    np.linspace(0,360,settings.phbins),
    amfr_conjstar,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
plt.xticks([0,60,120,180,240,300,360],[])
plt.yticks(fontsize=fs)
#plt.xlabel('Movement direction (deg)',fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
# plt.title('star-like walk',fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60,1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax6.transAxes
)
# plt.ylim([0,1.2*max(fr_mean)])
plt.ylim([0,5100])


# piecewise linear trajectory
# with open(pwl_traj_fname, 'rb') as f:
#         trajec = pickle.load(f)


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


ax_pwl_traj = fig.add_subplot(spec[4,0], adjustable='box', aspect="auto")
plt.plot(trajec[1][:int(part*2)],trajec[2][:int(part*2)],trajc, linewidth=1.5)
# plt.plot(trajec[1],trajec[2],'k')
ax_pwl_traj.set_aspect('equal', adjustable='box')
plt.axis('square')
plt.xticks([])
plt.yticks([])
# plt.xticks([0])
# plt.yticks([0])


bar_dist = 11 * settings.grsc
bar_ang = 60
plt.plot(
    [bar_dist * np.cos(2*np.pi*bar_ang/360), bar_dist * np.cos(2*np.pi*bar_ang/360) - bar_len, bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len], 
    [bar_dist * np.sin(2*np.pi*bar_ang/360), bar_dist * np.sin(2*np.pi*bar_ang/360), bar_dist * np.sin(2*np.pi*bar_ang/360)],
    color="red",
    lw=3
)


X_bgr,Y_bgr, _ = np.meshgrid(
    np.linspace(-2000,2000,4000),
    np.linspace(-2000,2000,4000),
    1
)
gr_bgr = grid_meanfr(X_bgr,Y_bgr,grsc=30, angle=0, offs=np.array([0,0]))
ax_pwl_traj.pcolor(X_bgr[:, :, 0],Y_bgr[:, :, 0],gr_bgr[:, :, 0], shading='auto')


# pwl firing rate
ax7 = fig.add_subplot(spec[4,1], adjustable='box', aspect="auto")
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
# ax7.spines['bottom'].set_visible(False)
# ax7.spines['left'].set_visible(False)
ax7.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
pwldata_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "pwldata.pkl"
)
conjpwl_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "conjunctive_pwl_N1024_nphi360_rmin0_rmax3_sigma0_kappa50_ratio1.npy"
            # "2C_conjunctive_pwl_N1024_nphi7200_rmin0_rmax3_sigma0_kappa50_ratio1.npy"
        )
)
if not os.path.isfile(pwldata_fname):
    # t, x, y, direc = traj(
    #     dt=dt_pwl,
    #     tmax=tmax_pwl,
    #     sp=settings.speed,
    #     init_dir=settings.init_dir,
    #     dphi=settings.dphi
    # )
    # trajec = pwliner(dt_pwl2, t, x, y, direc)
    t, x, y, direc = traj_pwl(
        settings.phbins_pwl,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(pwldata_fname), exist_ok=True)
    with open(pwldata_fname, 'wb') as f:
        pickle.dump((fr_mean, gr60), f)
else:
    with open(pwldata_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjpwl_afname, 'rb') as f:
        amfr_conjpwl = np.load(f)


plt.plot(np.linspace(0,360,settings.phbins), fr_mean, 'k', zorder = 10)
plt.plot(
    np.linspace(0,360,settings.phbins_pwl),
    # np.linspace(0,360,len(amfr_conjpwl)),
    amfr_conjpwl,
    color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
plt.xticks([0,60,120,180,240,300,360],[])
plt.yticks(fontsize=fs)
#plt.xlabel('Movement direction (deg)',fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
# plt.title('piece-wise linear walk',fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60,1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax7.transAxes
)
# plt.ylim([0,1.2*max(fr_mean)])
plt.ylim([0,5100])


# random walk trajectory
if not os.path.isfile(rw_traj_fname):
    trajec = traj(
        settings.dt,
        settings.tmax2,
        sp = settings.speed,
        dphi = settings.dphi
    )
    os.makedirs(os.path.dirname(rw_traj_fname), exist_ok=True)
    with open(rw_traj_fname, 'wb') as f:
        pickle.dump(trajec, f)
else:
    with open(rw_traj_fname, 'rb') as f:
        trajec = pickle.load(f)


ax_rw_traj = fig.add_subplot(spec[5,0], adjustable='box', aspect="auto")
plt.plot(trajec[1][:part],trajec[2][:part],trajc, linewidth=1.5)
ax_rw_traj.set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
plt.axis('square')


bar_dist = 1 * settings.grsc
bar_ang = 60
bar_len = 4 * settings.grsc
plt.plot(
    [bar_dist * np.cos(2*np.pi*bar_ang/360) + 2 * settings.grsc, bar_dist * np.cos(2*np.pi*bar_ang/360) + bar_len + 2 * settings.grsc], 
    [bar_dist * np.sin(2*np.pi*bar_ang/360), bar_dist * np.sin(2*np.pi*bar_ang/360)],
    color="red",
    lw=3
)


xtr,ytr,dirtr = trajec[1][:part], trajec[2][:part], trajec[3][:part]
tol = 5 * np.pi / 180
ax_rw_traj.pcolor(X_bgr[:, :, 0],Y_bgr[:, :, 0],gr_bgr[:, :, 0], shading='auto')


# rw firing rate
rwdata_fname = os.path.join(
    settings.loc,
    "conjunctive",
    "fig3",
    "rwdata.pkl"
)
conjrw_afname = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            "data",
            "analytical",
            "2D_conjunctive_rw_N1024_nphi360_T180000_dt0_01_v0_1_sigmatheta0_5_sigma0_kappa50_ratio1.npy"
        )
)
ax8 = fig.add_subplot(spec[5,1], adjustable='box', aspect="auto")
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
if not os.path.isfile(rwdata_fname):
    trajec = traj(
        settings.dt,
        settings.tmax2,
        sp = settings.speed,
        dphi = settings.dphi
    )
    direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=settings.propconj_i,
        kappa=settings.kappac_i,
        jitter=settings.jitterc_i
    )
    gr60 = get_hexsym(summed_fr, trajec)
    os.makedirs(os.path.dirname(rwdata_fname), exist_ok=True)
    with open(rwdata_fname, 'wb') as f:
        pickle.dump([fr_mean, gr60], f)
else:
    with open(rwdata_fname, 'rb') as f:
        fr_mean, gr60 = pickle.load(f)
with open(conjrw_afname, 'rb') as f:
        amfr_conjrw = np.load(f)


plt.plot(np.linspace(0,360,settings.phbins), fr_mean, 'k', zorder = 10)
plt.plot(
    np.linspace(0,360,settings.phbins),
    amfr_conjrw, color="darkgray",
    linestyle='-',
    linewidth=3,
    zorder=0
)
plt.xticks([0,60,120,180,240,300,360],fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel(r'Movement direction ($^\circ$)',fontsize=fs)
# plt.ylabel('Total firing \nrate (spk/s)', fontsize=fs)
plt.ylabel('Population\nfiring rate\n(spk/s)', fontsize=fs)
plt.text(
    hextextloc[0],
    hextextloc[1],
    'H = '+str(np.round(gr60,1))+" spk/s",
    fontsize=fs,
    horizontalalignment='center',
    verticalalignment='center',
    transform=ax8.transAxes
)
# plt.ylim([0,1.2*max(fr_mean)])
plt.ylim([0,5100])


###############################################################################
# Tweaking figure layout and saving                                           #
###############################################################################
plt.subplots_adjust(wspace = 0.60, hspace = 1.5)


ax_pos(ax9, 0, 0, 0.95, 0.95)
div_psearch = make_axes_locatable(ax9)
cax_psearch = div_psearch.append_axes('right', size='4.5%', pad=0.05)
cbar_psearch = fig.colorbar(pcolor_psearch, cax = cax_psearch, fraction=0.020, pad=0.04)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=fs)
# cbar.set_label('Hexasymmetry',fontsize=fs)
cbar_psearch.set_label("Hexasymmetry\n(spk/s)", fontsize=int(settings.fs))


# make trajectory plots larger
posstar = ax_star_traj.get_position()
pospwl = ax_pwl_traj.get_position()
posrw = ax_rw_traj.get_position()

pointsstar = posstar.get_points()
pointspwl = pospwl.get_points()
pointsrw = posrw.get_points()

mean_pointsstar = np.array(
    [(pointsstar[0][0] + pointsstar[1][0])/2,
    (pointsstar[0][1] + pointsstar[1][1])/2]
)
mean_pointspwl = np.array(
    [(pointspwl[0][0] + pointspwl[1][0])/2,
    (pointspwl[0][1] + pointspwl[1][1])/2]
)
mean_pointsrw = np.array(
    [(pointsrw[0][0] + pointsrw[1][0])/2,
    (pointsrw[0][1] + pointsrw[1][1])/2]
)

pointsstar -= mean_pointsstar
pointspwl -= mean_pointspwl
pointsrw -= mean_pointsrw

pointsstar = 2*pointsstar
pointspwl = 2*pointspwl
pointsrw = 2*pointsrw

pointsstar += mean_pointsstar
pointspwl += mean_pointspwl
pointsrw += mean_pointsrw

posstar.set_points(pointsstar)
pospwl.set_points(pointspwl)
posrw.set_points(pointsrw)

ax_star_traj.set_position(posstar)
ax_pwl_traj.set_position(pospwl)
ax_rw_traj.set_position(posrw)


# move polar subplot slightly to the left
posax2 = ax2.get_position()
pointsax2 = posax2.get_points()
mean_pointsax2 = np.array(
    [(pointsax2[0][0] + pointsax2[1][0])/2,
    (pointsax2[0][1] + pointsax2[1][1])/2]
)
pointsax2 -= mean_pointsax2
pointsax2[0][0] -= 0.02
pointsax2[1][0] -= 0.02
pointsax2 *= 0.9
pointsax2 += mean_pointsax2
posax2.set_points(pointsax2)
ax2.set_position(posax2)


# # make parameter seach subplot taller
# posax9 = ax9.get_position()
# pointsax9 = posax9.get_points()
# mean_pointsax9 = np.array(
#     [(pointsax9[0][0] + pointsax9[1][0])/2,
#     (pointsax9[0][1] + pointsax9[1][1])/2]
# )
# pointsax9 -= mean_pointsax9
# # pointsax9[0][1] *= 1.1
# pointsax9[1][1] *= 1.3
# pointsax9 += mean_pointsax9
# posax9.set_points(pointsax9)
# ax9.set_position(posax9)

# poscbar = cbar.get_position()
# pointscbar = poscbar.get_points()
# mean_pointscbar = np.array(
#     [(pointscbar[0][0] + pointscbar[1][0])/2,
#     (pointscbar[0][1] + pointscbar[1][1])/2]
# )
# pointscbar -= mean_pointscbar
# # pointscbar[0][1] *= 1.1
# pointscbar[1][1] *= 5.5
# pointscbar += mean_pointscbar
# poscbar.set_points(pointscbar)
# cbar.set_position(poscbar)


os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
plt.savefig(plot_fname,dpi=300)
