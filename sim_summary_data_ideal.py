import numpy as np
import matplotlib.pyplot as plt
from scipy import special, stats, interpolate
import numba
from joblib import Parallel, delayed
import utils.settings as settings
from functions.gridfcts import (
    traj, gen_offsets, traj_pwl, traj_star, gridpop_clustering, gridpop_conj, gridpop_repsupp
)
from utils.utils import convert_to_rhombus, adap_euler, get_hexsym as get_hexsym2, get_pathsym, grid_meanfr
import os
from tqdm import tqdm


def convert_to_square(xr, yr):
    return xr-1/np.sqrt(3)*yr, 2/np.sqrt(3)*yr


def map_to_rhombus(x, y, grsc=30):
    xr, yr = x-1/np.sqrt(3)*y, 2/np.sqrt(3)*y
    xr, yr = np.mod(xr, grsc), np.mod(yr, grsc)
    xmod, ymod = xr+0.5*yr, np.sqrt(3)/2*yr
    return xmod, ymod


@numba.njit(fastmath = True, parallel = True)
def grid_reduc(X, Y, grsc=30, angle=0, offs=np.array([0.,0.])):
    sc = 1/grsc * 2./np.sqrt(3)
    res = np.zeros((len(offs), np.shape(X)[0], np.shape(X)[1]))
    for i, off in enumerate(offs):
        res[i] = (1 + np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X[:, :, i]-(grsc*off[0]))+
                        2*np.pi*sc*np.cos(angle*np.pi/180)*(Y[:, :, i]-(grsc*off[1]))))*(1 + np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X[:, :, i]-(grsc*off[0]))+
               2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y[:, :, i]-(grsc*off[1]))))*(1 + np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X[:, :, i]-(grsc*off[0]))+
               2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y[:, :, i]-(grsc*off[1]))))
    return res # np.moveaxis(res, 0, -1)
        
def grid(X, Y, grsc=30, angle=0, offs=np.array([0.,0.]), rect=False):
    sc = 1/grsc * 2./np.sqrt(3)
    if rect:
        rec = lambda x: np.where(x>0, x, 0)
        if np.ndim(offs) == 1:
            return rec(np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X-grsc*offs[0])+2*np.pi*sc*np.cos(angle*np.pi/180)*(Y-grsc*offs[1]))) * rec(np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X-grsc*offs[0])+2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y-grsc*offs[1]))) * rec(np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X-grsc*offs[0])+2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y-grsc*offs[1])))
        else:
            assert len(offs[0]) == len(offs[1]), "ox and oy must have same length"
            return rec(
                np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X[:,:,None]-(grsc*offs[0])[None,None,:])+
                       2*np.pi*sc*np.cos(angle*np.pi/180)*(Y[:,:,None]-(grsc*offs[1])[None,None,:])))*\
                   rec(
                np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X[:,:,None]-(grsc*offs[0])[None,None,:])+
                       2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y[:,:,None]-(grsc*offs[1])[None,None,:])))*\
                   rec(
                np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X[:,:,None]-(grsc*offs[0])[None,None,:])+
                       2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y[:,:,None]-(grsc*offs[1])[None,None,:])))
    else:
        if np.size(offs) == 2:
            return (1 + np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X-grsc*offs[0])+2*np.pi*sc*np.cos(angle*np.pi/180)*(Y-grsc*offs[1]))) * (1 + np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X-grsc*offs[0])+2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y-grsc*offs[1]))) * (1 + np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X-grsc*offs[0])+2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y-grsc*offs[1])))
        else:
            assert len(offs[0]) == len(offs[1]), "ox and oy must have same length"
            return (1 +
                np.cos(2*np.pi*sc*np.sin(angle*np.pi/180)*(X[:,:,None]-(grsc*offs[0])[None,None,:])+
                       2*np.pi*sc*np.cos(angle*np.pi/180)*(Y[:,:,None]-(grsc*offs[1])[None,None,:])))*\
                   (1 +
                np.cos(2*np.pi*sc*np.sin((angle+60)*np.pi/180)*(X[:,:,None]-(grsc*offs[0])[None,None,:])+
                       2*np.pi*sc*np.cos((angle+60)*np.pi/180)*(Y[:,:,None]-(grsc*offs[1])[None,None,:])))*\
                   (1 +
                np.cos(2*np.pi*sc*np.sin((angle+120)*np.pi/180)*(X[:,:,None]-(grsc*offs[0])[None,None,:])+
                       2*np.pi*sc*np.cos((angle+120)*np.pi/180)*(Y[:,:,None]-(grsc*offs[1])[None,None,:])))


def gridpop(
    ox,
    oy,
    phbins,
    trajec,
    Amax = 1,
    grsc = 30,
    repsuppcell = False,
    repsupppop = False,
    tau_rep = 3.5,
    w_rep = 50,
    conj = False,
    propconj = 1.,
    kappa = 50,
    jitter=0,
    shufforient=False
):
    """
    simulate a population of grid cells for a complex 2d trajectory    
    """
    if type(ox)==int or type(ox)==float:
        N = 1
    else:
        N = len(ox)
    x, y, direc, t = trajec
    oxr,oyr = convert_to_rhombus(ox,oy)
    oxr,oyr = np.asarray(oxr), np.asarray(oyr)
    fr = np.zeros(np.shape(x))
    for n in range(N):
        if shufforient:
            ang = 360*np.random.rand()
            if repsuppcell:
                fr += adap_euler(Amax*grid(x,y,grsc=grsc,angle=ang,offs=(oxr[n],oyr[n])), t, tau_rep, w_rep)
            elif conj: # not used together at this stage
                if n<= int(propconj*N):
                    mu = np.mod(ang + jitter*np.random.randn(),360) #np.mod(np.random.randint(0,6,1) * 60 + jitter*np.random.randn(), 360)
                    grid(x,y,grsc=grsc,angle=ang,offs=(oxr[n],oyr[n]))
                    fr += Amax*grid(x,y,grsc=grsc,angle=ang,offs=(oxr[n],oyr[n])) * np.exp(kappa*np.cos(direc-np.pi/180.*mu))/special.i0(kappa) #(2*np.pi*special.i0(kappa))
                    # scheint nicht ok, immer noch normiert?
                else:
                    fr += Amax*grid(x,y,grsc=grsc,angle=ang,offs=(oxr[n],oyr[n]))
            else:
                fr += Amax*grid(x,y,grsc=grsc,angle=ang,offs=(oxr[n],oyr[n]))

        else:
            if repsuppcell:
                fr += adap_euler(Amax*grid(x,y,grsc=grsc,offs=(oxr[n],oyr[n])), t, tau_rep, w_rep)
            elif conj: # not used together at this stage
                if n<= int(propconj*N):
                    mu = np.mod(np.random.randint(0,6,1) * 60 + jitter*np.random.randn(), 360)
                    fr += Amax*grid(x,y,grsc=grsc,offs=(oxr[n],oyr[n])) * np.exp(kappa*np.cos(direc-np.pi/180.*mu))/special.i0(kappa) #(2*np.pi*special.i0(kappa))
                    # scheint nicht ok, immer noch normiert?
                else:
                    fr += Amax*grid(x,y,grsc=grsc,offs=(oxr[n],oyr[n]))
            else:
                fr += Amax*grid(x,y,grsc=grsc,angle=0, offs=(oxr[n],oyr[n]))

    if repsupppop:
        fr = adap_euler(fr, t, tau_rep, w_rep)

    #direc = np.mod(direc,2*np.pi)
    direc_binned = np.linspace(-np.pi,np.pi,phbins+1)
    #fr_mean2 = np.sum(fr[np.digitize(direc, direc_binned)])*phbins/len(fr)
    fr_mean = np.zeros(len(direc_binned)-1)
    for id in range(len(direc_binned)-1):
        ind = (direc>=direc_binned[id])*(direc<direc_binned[id+1])
        fr_mean[id] = np.sum(fr[ind])*phbins/len(fr)

    return fr_mean, np.abs(np.sum(fr * np.exp(-6j*direc)))/np.size(direc), np.abs(np.sum(np.exp(-6j*direc)))/np.size(direc), np.mean(fr)


def gridpop_meanfr(
    rmax = 90,
    Amax = 1,
    bins = 500,
    phbins = 360,
    mode = 'randi',
    N = 100,
    path = 'linear',
    dphi = 0.1,
    clsize = 0.25,
    meanoff = (0, 0),
    conj = False,
    propconj = 1,
    kappa = 50.,
    jitter = 0,
    kappacl = 10,
    repsuppcell = False,
    repsupppop = False,
    tau_rep = 5.,
    w_rep = 1.,
    speed = 10.
):
    if mode == 'clust':
        ox,oy = np.mod(np.linspace(meanoff[0]-clsize/2.,meanoff[0]+clsize/2.,N),1), np.mod(np.linspace(meanoff[1]-clsize/2.,meanoff[1]+clsize/2.,N),1)
        # just a linear stretch of phases, use meshgrid to get a small rhombus
    elif mode == 'clustcirc':
        # sample N cells from a circle of radius clsize around meanoff
        r = clsize/2.*np.sqrt(np.random.rand(N))
        phi = 2*np.pi*np.random.rand(N)
        ox,oy = np.mod(meanoff[0] + r*np.cos(phi),1), np.mod(meanoff[1] + r*np.sin(phi),1)
    elif mode == 'clustvm':
        ox, oy = np.random.vonmises(2*np.pi*(meanoff[0]-0.5), kappacl, N)/2./np.pi + 0.5, np.random.vonmises(2*np.pi*(meanoff[1]-0.5), kappacl, N)/2./np.pi + 0.5
    elif mode == 'uniform':
        ox,oy = np.meshgrid(np.linspace(0,1,int(np.sqrt(N)),endpoint=False), np.linspace(0,1,int(np.sqrt(N)),endpoint=False))
        ox = ox.reshape(1,-1)[0]
        oy = oy.reshape(1,-1)[0]
        N = int(np.sqrt(N))**2
    elif mode == 'randi':
        ox,oy = np.random.rand(N), np.random.rand(N)
    else:
        ox,oy = np.array([meanoff[0]]), np.array([meanoff[1]])

    oxr, oyr = convert_to_rhombus(ox, oy)
    if path == 'linear':
        star_offs = np.random.rand(2)
        offxr, offyr = convert_to_rhombus(star_offs[0], star_offs[1])
        if N == 1:
            r, phi = np.meshgrid(np.linspace(0, rmax, bins), np.linspace(0, 360, phbins, endpoint=False))
            X, Y = r*np.cos(phi*np.pi/180) + offxr * settings.grsc, r*np.sin(phi*np.pi/180) + offyr * settings.grsc
            # X, Y = r*np.cos(phi*np.pi/180), r*np.sin(phi*np.pi/180)
            
            grids = Amax * grid_reduc(X,Y,offs=(oxr,oyr), rect=True)
            grids2 = grids.copy()
            """
            plt.figure()
            slope = np.tan(np.pi/3)
            for yy in np.arange(-300,301,np.sqrt(3)/2*30):
                plt.plot([-300,300],[yy,yy],'k')
            for xx in np.arange(-500,501,30):
                plt.plot([-2/np.sqrt(3)*300+xx,2/np.sqrt(3)*300+xx],[slope*-2/np.sqrt(3)*300,slope*2/np.sqrt(3)*300],'k')
            plt.scatter(X,Y,zorder=3,s=1)
            plt.axis([-rmax,rmax,-rmax,rmax])
            plt.xlabel('x (cm)')
            plt.ylabel('y (cm)')
            """

            # how does meanfr depend on bins ... Naomi showed that the different peaks in meanfr seem to have different heights for higher number of bins??
        else:
            r, phi, indoff = np.meshgrid(np.linspace(0, rmax, bins), np.linspace(0, 360, phbins, endpoint=False), np.arange(len(ox)))
            X, Y = r*np.cos(phi*np.pi/180) + offxr * settings.grsc, r*np.sin(phi*np.pi/180) + offyr * settings.grsc

            # calculate integrated firing rate as a function of the movement angle
            grids = Amax * grid_reduc(X, Y, offs=np.vstack((oxr, oyr)).T)
            grids = np.moveaxis(grids, 0, -1) #[:, :, 0, :]
            grids2 = grids.copy()

        if conj: # add head-direction tuning if wanted
            Nconj = int(propconj*N)
            #mu = np.random.randint(0,6,Nconj) * 60 # a multiple of 60deg
            mu = np.mod(np.random.randint(0, 6, Nconj) * 60 + jitter*np.random.randn(Nconj), 360)
            vonmi = np.exp(kappa*np.cos(np.pi/180*(np.linspace(0, 360, phbins)[:, None]-mu[None, :])))/special.i0(kappa)# (2*np.pi*special.i0(kappa))
            if N == 1 and Nconj == 1:
                grids = grids*vonmi
            else:
                grids[:, :, :Nconj] = grids[:, :, :Nconj]*vonmi[:, None, :] # change the first Nconj cells
        if repsuppcell:
            #speed = 1. # grid scale/s
            #tau_rep = 1. #s
            #w_rep = 3.
            #s = grids[0,:,0]

            tt = np.linspace(0,rmax/speed,bins)
            #aa = np.zeros(np.shape(grids))
            for idir in range(phbins):
                if N == 1:
                    v = adap_euler(grids[idir,:], tt, tau_rep, w_rep)
                    grids[idir,:] = v
                else:
                    for ic in range(N):
                        v = adap_euler(grids[idir, :, ic], tt, tau_rep, w_rep)
                        grids[idir, :, ic] = v
                        #aa[idir,:,ic] = a

        if repsupppop:
            tt = np.linspace(0,rmax/speed,bins)
            meanpop_base = np.sum(grids,axis=2)
            meanpop = np.sum(grids,axis=2)
            for idir in range(phbins):
                v = adap_euler(meanpop[idir,:],tt,tau_rep,w_rep)
                meanpop[idir,:] = v
            meanfr = np.sum(meanpop, axis=1)/bins
            meanfr_base = np.sum(meanpop_base, axis=1)/bins
            meanfr = meanfr/np.mean(meanfr)*np.mean(meanfr_base)

        else:
            if N==1:
                meanfr_base = np.sum(grids2,axis=1)/bins
                meanfr = np.sum(grids,axis=1)/bins
                meanfr = meanfr/np.mean(meanfr)*np.mean(meanfr_base)
            else:
                meanfr_base = np.sum(np.sum(grids2,axis=1)/bins,axis=1)
                meanfr = np.sum(np.sum(grids,axis=1)/bins,axis=1)
                meanfr = meanfr/np.mean(meanfr)*np.mean(meanfr_base)
        gr2 = np.sum(grids, axis=2)
        gr60 = np.abs(np.sum(gr2 * np.exp(-6j*np.pi/180*phi[:, :, 0])))/np.size(gr2)
        gr60_path = np.abs(np.sum(np.exp(-6j*np.pi/180*phi[:, :, 0])))/np.size(gr2)
        gr0 = np.sum(gr2)/np.size(gr2)

    else: # random walk trajectory
        tmax = settings.tmax
        dt = settings.dt
        trajec = traj(dt,tmax)
        if False: #repsuppcell or repsupppop:
            meanfr_base = gridpop(oxr,oyr,phbins,trajec,repsuppcell=False,repsupppop=False,tau_rep=tau_rep,w_rep=w_rep, conj = conj, propconj=propconj, kappa=kappa, jitter=jitter)
            meanfr, gr60, gr60_path, gr0 = gridpop(oxr,oyr,phbins,trajec,repsuppcell=repsuppcell,repsupppop=repsupppop,tau_rep=tau_rep,w_rep=w_rep, conj = conj, propconj=propconj, kappa=kappa, jitter=jitter)
            meanfr = meanfr/np.mean(meanfr)*np.mean(meanfr_base)
        else:
            meanfr, gr60, gr60_path, gr0 = gridpop(oxr,oyr,phbins,trajec,repsuppcell=repsuppcell,repsupppop=repsupppop,tau_rep=tau_rep,w_rep=w_rep, conj = conj, propconj=propconj, kappa=kappa, jitter=jitter)

    return gr60, gr60_path, gr0, gr60#gr90, p60, p90

def gridpop_conjcompare(ox, oy, phbins=360, Amax = 1, grsc = 30, dx = 1., propconj = 0.66, kappa = 50, jitter=10):
    """
    simulate the summed activity G(x,y,phi) of a population of grid cells for a part of space (x,y)
    and the movement angle phi, trajectories can later sample the summed grid-cell
    signal from this part of space. We can exploit that G(x,y,phi) is periodic in x and y
    (with the grid scale) and periodic in phi (with 2 pi).
    """
    if type(ox)==int or type(ox)==float:
        N = 1
    else:
        N = len(ox)
    oxr,oyr = convert_to_rhombus(ox,oy)
    x,y = np.meshgrid(np.arange(0,grsc,dx), np.arange(0,grsc,dx))
    xr,yr = convert_to_rhombus(x,y)

    # G is 4-dim at first, number of cells as extra dimension
    if N==1:
        G1 = Amax*grid(xr, yr, grsc=grsc, angle=0, offs=(oxr,oyr))[:,:,None]
    else:
        G1 = Amax*grid(xr, yr, grsc=grsc, angle=0, offs=(oxr,oyr)) # G1(x,y,N)
    direc_bins = np.pi/180 * np.arange(0,360,360/phbins) # dphi = 1 deg

    # include conjunctive HD tuning
    if N==6:
        mu = np.mod(np.arange(0,360,60) + jitter*np.random.randn(N), 360) # different for different cells
    else:
        mu = np.mod(np.random.randint(0,6,N) * 60 + jitter*np.random.randn(N), 360) # different for different cells
    rv = stats.vonmises(kappa,loc=np.pi/180*mu)
    h = 2*np.pi * np.array([rv.pdf(d) for d in direc_bins]).T
    G2 = G1[:,:,:,None] * h[None,None,:,:]

    # exclude conjunctive HD tuning
    h2 = np.ones(len(direc_bins))
    G3 = G1[:,:,:,None] * h2[None,None,None,:]

    G2sum = np.sum(G2, axis=2)
    G3sum = np.sum(G3, axis=2)
    # to get back G1:
    # G1back = np.sum(G3sum,axis=2)*2*np.pi/phbins
    return G2sum, G3sum

def apply_traj(G, trajec, grsc=30, dx=1, dt=0.1, phbins=360):
    # map trajectory to rhombus and read firing rate from G
    t, x, y, di = trajec
    """
    plt.figure()
    plt.plot(xr,yr,'.-')
    plt.plot(xr+30,yr,'.-')
    plt.plot(xr+15,yr+np.sqrt(3)/2 * 30,'.-')
    """
    xr, yr = convert_to_square(*map_to_rhombus(x, y, grsc=grsc))

    xedge, yedge = np.arange(0, grsc+dx, dx), np.arange(0, grsc+dx, dx)
    diredge = np.pi/180 * np.linspace(-180, 180, phbins+1) # dphi = 1 deg

    # meanfr = np.sum(np.sum(G * H,axis=0),axis=0)*dx*dx*40 # where does 40 come from?
    xd, yd, dird = np.digitize(xr, xedge)-1, np.digitize(yr, yedge)-1, np.digitize(di, diredge)-1
    fr = G[xd, yd, np.where(dird==360, 0, dird)]
    return fr, di

def get_hexsym(fr, phi):
    gr60 = np.abs(np.sum(fr * np.exp(-6j*phi)))/np.size(fr)
    gr60_path = np.abs(np.sum(np.exp(-6j*phi[1:])))/len(phi[1:])
    gr0 = np.sum(fr)/np.size(fr)
    return gr60, gr60_path, gr0


###############################################################################
# Parameters & array initialisation                                           #
###############################################################################
rep = settings.rep

# cellular
Ncells = settings.N


# conjunctive
propconj = settings.propconj_i
kappa = settings.kappac_i
jitter = settings.jitterc_i

# clustering
kappacl = settings.kappa_si
meanoff = settings.meanoff


# repsupp
phbins = settings.phbins
dx = 1
tau_rep = settings.tau_rep
w_rep = settings.w_rep


# initialise hex arrays. (gr60, gr60_path, gr0)
h_cl_star = np.zeros((3, rep))
h_cl_rw = np.zeros((3, rep))
h_cl_pl = np.zeros((3, rep))
h_conj_star = np.zeros((3, rep))
h_conj_rw = np.zeros((3, rep))
h_conj_pl = np.zeros((3, rep))
h_rep_star = np.zeros((3, rep))
h_rep_star_noreset = np.zeros((3, rep))
h_rep_rw = np.zeros((3, rep))
h_rep_pl = np.zeros((3, rep))


###############################################################################


# Gconj1, Gnoconj1 = gridpop_conjcompare(
#     ox2,
#     oy2,
#     phbins,
#     Amax = settings.amax,
#     grsc = settings.grsc,
#     dx = dx,
#     propconj = propconj, 
#     kappa = kappa,
#     jitter = jitter
# )
# Gconjclust, Gclust = gridpop_conjcompare(
#     ox,
#     oy,
#     phbins,
#     grsc = settings.grsc,
#     dx = dx,
#     propconj = propconj,
#     kappa = kappa,
#     jitter=jitter
# )


def mfunc(i):
    # clustering offsets
    ox, oy = gen_offsets(N=settings.N, kappacl=kappacl)
    oxr, oyr = convert_to_rhombus(ox, oy)


    # uniform offsets
    # ox2, oy2 = gen_offsets(N=settings.N, kappacl=0.)
    # # ox2 = ox2.reshape(1,-1)[0]
    # # oy2 = oy2.reshape(1,-1)[0]
    # ox2r, oy2r = convert_to_rhombus(ox2, oy2)

    ox2,oy2 = np.meshgrid(
            np.linspace(0,1,int(np.sqrt(settings.N)),endpoint=False),
            np.linspace(0,1,int(np.sqrt(settings.N)),endpoint=False)
        )
    ox2r = ox2.reshape(1,-1)[0]
    oy2r = oy2.reshape(1,-1)[0]
    ################
    # random walks #
    ################
    trajec = traj(settings.dt, settings.tmax)
    # fr_conj, phi1 = apply_traj(
    #     Gconj1, trajec, grsc=settings.grsc, dx=dx, dt=settings.dt, phbins=phbins
    # )

    # conjunctive
    direc_binned, fr_mean, _, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=propconj,
        kappa=kappa,
        jitter=jitter
    )
    h_conj_rw = (get_hexsym2(summed_fr, trajec), get_pathsym(trajec), np.mean(fr_mean))


    # fr_clust, phi2 = apply_traj(
    #     Gclust, trajec, grsc=settings.grsc, dx=dx, dt=settings.dt, phbins=phbins
    # )
    # # h_conj_rw = get_hexsym(fr_conj, trajec[2])
    # h_cl_rw = get_hexsym(fr_clust, trajec[2])

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N = settings.N,
        grsc = settings.grsc,
        phbins = settings.phbins,
        traj = trajec,
        oxr = oxr,
        oyr = oyr
    )
    h_cl_rw = (get_hexsym2(summed_fr, trajec), get_pathsym(trajec), np.mean(fr_mean))

    # repsupp
    direc_binned, fr_mean, _, summed_fr = gridpop_repsupp(
        settings.N,
        settings.grsc,
        settings.phbins,
        traj=trajec,
        oxr=ox2r,
        oyr=oy2r,
        tau_rep=tau_rep,
        w_rep=w_rep
    )
    h_rep_rw = (get_hexsym2(summed_fr, trajec), get_pathsym(trajec), np.mean(fr_mean))

    # random walk
    h_rep_rw = gridpop_meanfr(
        rmax=settings.rmax,
        bins=settings.bins,
        mode='randi',
        path='',
        N=Ncells,
        conj=False,
        repsuppcell = True,
        tau_rep=tau_rep,
        w_rep=w_rep
    )[:3]

    ###################
    # star-like walks #
    ###################
    # h_cl_star = gridpop_meanfr(
    #     rmax=settings.rmax,
    #     bins=settings.bins,
    #     mode='clustvm',
    #     path='linear',
    #     N=Ncells,
    #     kappacl=kappacl,
    #     conj=False,
    #     repsuppcell = False
    # )[:3]
    # h_conj_star = gridpop_meanfr(
    #     rmax=settings.rmax,
    #     bins=settings.bins,
    #     mode='randi',
    #     path='linear',
    #     N=Ncells,
    #     conj=True,
    #     propconj = propconj,
    #     kappa = kappa,
    #     jitter = jitter,
    #     repsuppcell = False
    # )[:3]



    h_rep_star_noreset = gridpop_meanfr(
        rmax=settings.rmax,
        bins=settings.bins,
        mode='randi',
        # mode='uniform',
        path='linear',
        N=Ncells,
        conj=False,
        repsuppcell = True,
        tau_rep=tau_rep,
        w_rep=w_rep
    )[:3]

    # tt = np.linspace(0,settings.rmax/settings.speed,settings.bins)
    # r,phi,indoff = np.meshgrid(np.linspace(0,settings.rmax,settings.bins), np.linspace(0,2 * np.pi,settings.phbins, endpoint=False), np.arange(len(ox2)))
    # X,Y = r*np.cos(phi), r*np.sin(phi)
    # grids = settings.amax * grid_meanfr(X,Y,offs=(ox3r,oy3r))
    # grids2 = grids.copy()
    # for idir in range(settings.phbins):
    #     for ic in range(settings.N):
    #         v = adap_euler(grids[idir, :, ic],tt,settings.tau_rep,settings.w_rep)
    #         grids[idir, :, ic] = v

    # meanfr = np.sum(np.sum(grids,axis=1)/settings.bins)
    # gr2 = np.sum(grids, axis=2)
    # gr60 = np.abs(np.sum(gr2 * np.exp(-6j*phi[:, :, 0])))/np.size(gr2)
    # gr60_path = np.abs(np.sum(np.exp(-6j*np.pi/180*phi[:, :, 0])))/np.size(gr2)

    # h_rep_star_noreset = (gr60, gr60_path, meanfr)

    ###

    star_offs = np.random.rand(2)
    trajec_star = traj_star(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed,
        offset=star_offs
    )


    direc_binned, meanfr_star_noreset, _, summed_fr = gridpop_repsupp(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=ox2r,
        oyr=oy2r,
        tau_rep=tau_rep,
        w_rep=w_rep
    )
    h_rep_star = (get_hexsym2(summed_fr, trajec_star), get_pathsym(trajec_star), np.mean(meanfr_star_noreset[~np.isnan(meanfr_star_noreset).any()]))


    direc_binned, meanfr_star_noreset, _, summed_fr = gridpop_conj(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=ox2r,
        oyr=oy2r,
        propconj=propconj,
        kappa=kappa,
        jitter=jitter
    )
    h_conj_star = (get_hexsym2(summed_fr, trajec_star), get_pathsym(trajec_star), np.mean(meanfr_star_noreset[~np.isnan(meanfr_star_noreset).any()]))


    direc_binned, meanfr_star_noreset, _, summed_fr = gridpop_clustering(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_star = (get_hexsym2(summed_fr, trajec_star), get_pathsym(trajec_star), np.mean(meanfr_star_noreset[~np.isnan(meanfr_star_noreset).any()]))


    #####################
    # piece-wise linear #
    #####################
    trajec_pl = traj_pwl(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )

    # conjunctive
    direc_binned, fr_mean, _, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec_pl,
        ox2r,
        oy2r,
        propconj=propconj,
        kappa=kappa,
        jitter=jitter
    )
    h_conj_pl = (get_hexsym2(summed_fr, trajec_pl), get_pathsym(trajec_pl), np.mean(fr_mean[~np.isnan(fr_mean).any()]))

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N = settings.N,
        grsc = settings.grsc,
        phbins = settings.phbins,
        traj = trajec_pl,
        oxr = oxr,
        oyr = oyr
    )
    h_cl_pl = (get_hexsym2(summed_fr, trajec_pl), get_pathsym(trajec_pl), np.mean(fr_mean))

    # repsupp
    direc_binned, fr_mean, _, summed_fr = gridpop_repsupp(
        settings.N,
        settings.grsc,
        settings.phbins,
        traj=trajec_pl,
        oxr=ox2r,
        oyr=oy2r,
        tau_rep=tau_rep,
        w_rep=w_rep
    )
    h_rep_pl = (get_hexsym2(summed_fr, trajec_pl), get_pathsym(trajec_pl), np.mean(fr_mean))

    return h_conj_star, h_conj_rw, h_conj_pl, h_cl_star, h_cl_rw, h_cl_pl, h_rep_star, h_rep_rw, h_rep_pl, h_rep_star_noreset


alldata = Parallel(
    n_jobs=-1, verbose=100)(delayed(mfunc)(i) for i in tqdm(range(rep))
)
alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)
h_conj_star, h_conj_rw, h_conj_pl, h_cl_star, h_cl_rw, h_cl_pl, h_rep_star, h_rep_rw, h_rep_pl, h_rep_star_noreset = alldata


# fig = plt.figure(figsize=(12, 12))
# plt.bar([0, 1, 2, 3.5, 4.5, 5.5, 7, 8, 9],
#         [np.mean(h_conj_star[0, :]), np.mean(h_conj_pl[0, :]), np.mean(h_conj_rw[0, :]),
#           np.mean(h_cl_star[0, :]), np.mean(h_cl_pl[0, :]), np.mean(h_cl_rw[0, :]),
#           np.mean(h_rep_star[0, :]), np.mean(h_rep_pl[0, :]), np.mean(h_rep_rw[0, :])], color='b')
# plt.plot(np.zeros(rep), h_conj_star[0, :], 'k.')
# plt.plot(np.ones(rep), h_conj_pl[0, :], 'k.')
# plt.plot(2*np.ones(rep), h_conj_rw[0, :], 'k.')
# plt.plot(3.5*np.ones(rep), h_cl_star[0, :], 'k.')
# plt.plot(4.5*np.ones(rep), h_cl_pl[0, :], 'k.')
# plt.plot(5.5*np.ones(rep), h_cl_rw[0, :], 'k.')
# plt.plot(7*np.ones(rep), h_rep_star[0, :], 'k.')
# plt.plot(8*np.ones(rep), h_rep_pl[0, :], 'k.')
# plt.plot(9*np.ones(rep), h_rep_rw[0, :], 'k.')
# plt.xticks([0, 1, 2, 3.5, 4.5, 5.5, 7, 8, 9], ['star', 'p-l \n Conjunctive', 'rand',
#                                                 'star', 'p-l \n Clustering', 'rand',
#                                                 'star', 'p-l \n Repetition suppression', 'rand'])
# plt.bar([0, 1, 2, 3.5, 4.5, 5.5, 7, 8, 9],
#         [np.mean(h_conj_star[1, :]*h_conj_star[2, :]), np.mean(h_conj_pl[1, :]*h_conj_pl[2, :]), np.mean(h_conj_rw[1, :]*h_conj_rw[2, :]),
#           np.mean(h_cl_star[1, :]*h_cl_star[2, :]), np.mean(h_cl_pl[1, :]*h_cl_pl[2, :]), np.mean(h_cl_rw[1, :]*h_cl_rw[2, :]),
#           np.mean(h_rep_star[1, :]*h_rep_star[2, :]), np.mean(h_rep_pl[1, :]*h_rep_pl[2, :]), np.mean(h_rep_rw[1, :]*h_rep_rw[2, :])], color='b', edgecolor='k')
# plt.ylabel('H')
# plt.yscale('log')

# os.makedirs(os.path.dirname(os.path.join(settings.loc, "summary", 'summary_figure_ideal.png')), exist_ok=True)
# plt.savefig(os.path.join(settings.loc, "summary", 'summary_figure_ideal.png'), dpi=300)
# plt.close(fig)

# same figure for idealistic parameter choices

data = np.array([h_conj_star[0, :], h_conj_star[1, :], h_conj_star[2, :], 
                 h_conj_pl[0, :], h_conj_pl[1, :], h_conj_pl[2, :], 
                 h_conj_rw[0, :], h_conj_rw[1, :], h_conj_rw[2, :],
                 h_cl_star[0, :], h_cl_star[1, :], h_cl_star[2, :], 
                 h_cl_pl[0, :], h_cl_pl[1, :], h_cl_pl[2, :],
                 h_cl_rw[0, :], h_cl_rw[1, :], h_cl_rw[2, :],
                 h_rep_star[0, :], h_rep_star[1, :], h_rep_star[2, :],
                 h_rep_pl[0, :], h_rep_pl[1, :], h_rep_pl[2, :],
                 h_rep_rw[0, :], h_rep_rw[1, :], h_rep_rw[2, :],
                 h_rep_star_noreset[0, :], h_rep_star_noreset[1, :], h_rep_star_noreset[2, :]])
np.save(os.path.join(settings.loc, "summary",'summary_bar_plot_3hyp3traj_ideal'), data)
