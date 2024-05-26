import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pickle
import settings
import os
from scipy.stats import multivariate_normal
from scipy.signal import convolve as convolve_sc
from joblib import Parallel, delayed
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.utils import ax_pos
from scipy.stats import vonmises, mannwhitneyu

###############################################################################
# Parameters                                                                  #
###############################################################################
n = 100
randfield = np.random.rand(n, n)
saveloc = os.path.join(
    settings.loc,
    "clustering",
    "fig4",
    "randfield.png"
)


###############################################################################
# Functions                                                                   #
###############################################################################
def convert_to_rhombus(x, y):
    return x+0.5*y, np.sqrt(3)/2*y


def circdiff(vec1, vec2, scale):
    output = np.array(
        [np.mod(vec1-vec2, scale), np.mod(vec2-vec1, scale)]
    ).min(axis=0)
    return output


def circmean(vec):
    return np.mod(np.angle(sum(np.exp(1j*vec)) / len(vec)), 2*np.pi)


def pairwise_distance(mat, gridx, gridy):
    mat_lin = np.reshape(mat, (1, -1))[0]
    m1, m2 = np.meshgrid(mat_lin, mat_lin)
    xlin = np.reshape(gridx, (1, -1))[0]
    x1, x2 = np.meshgrid(xlin, xlin)
    ylin = np.reshape(gridy, (1, -1))[0]
    y1, y2 = np.meshgrid(ylin, ylin)
    xdiff = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    phdiff = circdiff(m1, m2)

    dx = 5
    bins = np.arange(0, 80, dx)
    meandiff = np.zeros(len(bins)-1)
    stddiff = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        meandiff[i] = np.mean(phdiff[(xdiff > bins[i]) * (xdiff < bins[i+1])])
        stddiff[i] = np.std(phdiff[(xdiff > bins[i]) * (xdiff < bins[i+1])])

    return xdiff, phdiff, meandiff, stddiff, bins


###############################################################################
# Plotting                                                                    #
###############################################################################
x, y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
X = np.dstack((x, y))
mu = np.zeros(2)
sigma = np.array([[9, 0], [0, 9]])
rv = multivariate_normal(mu, sigma)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(randfield)
plt.subplot(2, 2, 2)
plt.imshow(rv.pdf(X))
plt.subplot(2, 2, 3)
plt.imshow(convolve_sc(randfield, rv.pdf(X), 'same'))
plt.close()


# circular mean for certain radius
gridx, gridy = np.meshgrid(np.arange(n), np.arange(n))
res = np.zeros(np.shape(randfield))
radius = 20
for i in range(np.shape(randfield)[0]):
    for j in range(np.shape(randfield)[1]):
        dist = np.sqrt((gridx-i)**2 + (gridy-j)**2)
        res[i, j] = circmean(2*np.pi*randfield[dist < radius])
res = res/2/np.pi


def grid2(
    X,
    Y,
    sc=2./np.sqrt(3),
    angle=0,
    offs=np.array([0, 0])
):
    def rec(x): return np.where(x > 0, x, 0)
    if np.size(offs) == 2:
        return rec(
            np.cos(
                2 * np.pi * sc * np.sin(angle * np.pi / 180) * (X - offs[0])
                + 2 * np.pi * sc * np.cos(angle * np.pi / 180) * (Y - offs[1]))
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 60) * np.pi / 180)
                * (X - offs[0]) + 2 * np.pi * sc
                * np.cos((angle + 60) * np.pi / 180) * (Y - offs[1]))
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 120) * np.pi / 180)
                * (X - offs[0]) + 2 * np.pi * sc
                * np.cos((angle + 120) * np.pi / 180) * (Y - offs[1]))
        )
    else:
        assert len(offs[0]) == len(offs[1]), "ox and oy must have same length"
        return rec(
            np.cos(
                2 * np.pi * sc * np.sin(angle * np.pi / 180)
                * (X[:, :, None] - (offs[0])[None, None, :]) + 2 * np.pi * sc
                * np.cos(angle * np.pi / 180)
                * (Y[:, :, None] - (offs[1])[None, None, :])
            )
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 60) * np.pi / 180)
                * (X[:, :, None] - (offs[0])[None, None, :]) + 2 * np.pi * sc
                * np.cos((angle + 60) * np.pi / 180)
                * (Y[:, :, None] - (offs[1])[None, None, :])
            )
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 120) * np.pi / 180) *
                (X[:, :, None] - (offs[0])[None, None, :]) + 2 * np.pi * sc
                * np.cos((angle + 120) * np.pi / 180)
                * (Y[:, :, None] - (offs[1])[None, None, :])
            )
        )


def grid3(
    X,
    Y,
    Z,
    sc=2./np.sqrt(3),
    angle=0,
    offs=np.array([0, 0, 0])
):
    def rec(x): return np.where(x > 0, x, 0)
    if np.size(offs) == 3:
        ret = rec(
            np.cos(
                2 * np.pi * sc * np.sin(angle * np.pi / 180) * (X - offs[0])
                + 2 * np.pi * sc * np.cos(angle * np.pi / 180) * (Y - offs[1])
            )
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 60) * np.pi / 180) *
                (X - offs[0]) + 2 * np.pi * sc * np.cos(
                    (angle + 60) * np.pi / 180) * (Y - offs[1])
            )
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 120) * np.pi / 180) *
                (X - offs[0]) + 2 * np.pi * sc * np.cos(
                    (angle + 120) * np.pi / 180) * (Y - offs[1])
            )
        )
        return ret
    else:
        assert len(offs[0]) == len(offs[1]), "ox and oy must have same length"
        ret = rec(
            np.cos(
                2 * np.pi * sc * np.sin(angle * np.pi / 180) *
                (X[:, :, None] - (offs[0])[None, None, :]) + 2 * np.pi * sc
                * np.cos(angle * np.pi / 180) *
                (Y[:, :, None] - (offs[1])[None, None, :])
            )
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 60) * np.pi / 180) *
                (X[:, :, None] - (offs[0])[None, None, :]) + 2 * np.pi * sc
                * np.cos((angle + 60) * np.pi / 180)
                * (Y[:, :, None] - (offs[1])[None, None, :])
            )
        ) * rec(
            np.cos(
                2 * np.pi * sc * np.sin((angle + 120) * np.pi / 180) *
                (X[:, :, None] - (offs[0])[None, None, :]) + 2 * np.pi * sc
                * np.cos((angle + 120) * np.pi / 180)
                * (Y[:, :, None] - (offs[1])[None, None, :])
            )
        )
        return ret


def create_randomfield_2D(n, mode='grid'):
    # create 2D random field with grid AC
    # n = 100
    rand_ph_help = 2*np.pi*np.random.rand(n, n)  # draw phase of complex number
    # draw phase of complex number
    rand_ph_help2 = 2*np.pi*np.random.rand(n, n)
    randfield1 = np.cos(rand_ph_help)
    randfield1_2 = np.cos(rand_ph_help2)
    randfield2 = np.sin(rand_ph_help)
    randfield2_2 = np.sin(rand_ph_help2)
    x, y = np.meshgrid(
        np.linspace(0, 3, n),
        np.linspace(0, 3, n))  # (3mm)**3 voxel

    part = 1
    gridx, gridy = np.meshgrid(
        np.linspace(-1.5 / part, 1.5 / part, int(n / part)),
        np.linspace(-1.5 / part, 1.5 / part, int(n / part)))
    X = np.array(list(zip(np.reshape(gridx, (1, -1))
                 [0], np.reshape(gridy, (1, -1))[0])))
    if mode == 'gauss':
        mu = np.zeros(2)
        sigma = 0.03
        sig = np.array([[sigma, 0], [0, sigma]])**2
        rv = sc.stats.multivariate_normal(mu, sig)
        kernel = np.reshape(rv.pdf(X), (int(n/part), int(n/part)))
    elif mode == 'grid':
        grsc = 0.3
        kernel = np.reshape(
            grid2(
                X[:, 0],
                X[:, 1],
                sc=1 / grsc * 2 / np.sqrt(3)),
            (int(n / part),
             int(n / part)))
        kernel[np.sqrt(gridx**2 + gridy**2) > 1.5*grsc] = 0
        kernel = kernel/np.sum(kernel)/(3/(n-1))**2
    else:
        raise Exception('What are you doing?')
    # plt.figure()
    # plt.imshow(kernel[:,:,5])

    res_re = sc.signal.convolve(
        randfield1, kernel, 'same')  # use valid instead
    res_re_2 = sc.signal.convolve(randfield1_2, kernel, 'same')
    res_im = sc.signal.convolve(randfield2, kernel, 'same')
    res_im_2 = sc.signal.convolve(randfield2_2, kernel, 'same')
    ph1 = (np.angle(res_re + 1j*res_im) + np.pi)/2/np.pi
    ph2 = (np.angle(res_re_2 + 1j*res_im_2) + np.pi)/2/np.pi
    return ph1, ph2, x, y, kernel


def create_randomfield_3D(n, mode='grid'):
    # create 2D random field with grid AC
    # draw phase of complex number
    rand_ph_help = 2*np.pi*np.random.rand(n, n, n)
    # draw phase of complex number
    rand_ph_help2 = 2*np.pi*np.random.rand(n, n, n)

    randfield1 = np.cos(rand_ph_help)
    randfield1_2 = np.cos(rand_ph_help2)

    randfield2 = np.sin(rand_ph_help)
    randfield2_2 = np.sin(rand_ph_help2)

    x, y, z = np.meshgrid(
        np.linspace(0, 3, n),
        np.linspace(0, 3, n),
        np.linspace(0, 3, n))  # (3mm)**3 voxel

    part = 1
    gridx, gridy, gridz = np.meshgrid(
        np.linspace(-1.5/part, 1.5/part, int(n/part)),
        np.linspace(-1.5/part, 1.5/part, int(n/part)),
        np.linspace(-1.5/part, 1.5/part, int(n/part))
    )

    X = np.array(
        list(zip(
            np.reshape(gridx, (1, -1))[0],
            np.reshape(gridy, (1, -1))[0],
            np.reshape(gridz, (1, -1))[0]
        ))
    )
    sigma = 0.03
    grsc = 0.3

    if mode == "none":
        ph1 = (np.angle(randfield1 + 1j*randfield2) + np.pi)/2/np.pi
        ph2 = (np.angle(randfield1_2 + 1j*randfield2_2) + np.pi)/2/np.pi
        kernel = np.zeros((int(n/part), int(n/part), int(n/part)))
    else:
        if mode == 'gauss':
            mu = np.zeros(3)
            sig = np.eye(3) * sigma
            rv = sc.stats.multivariate_normal(mu, sig**2)
            kernel = np.reshape(
                rv.pdf(X),
                (int(n / part),
                 int(n / part),
                 int(n / part)))
        elif mode == 'grid':
            kernel = np.zeros((int(n/part), int(n/part), int(n/part)))
            ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            means = zip(grsc * np.cos(ang), grsc * np.sin(ang), np.zeros(6))
            sig = np.eye(3) * sigma
            rv = sc.stats.multivariate_normal([0, 0, 0], sig**2)
            kernel += np.reshape(rv.pdf(X), (int(n/part),
                                 int(n/part), int(n/part)))
            for mu in means:
                rv = sc.stats.multivariate_normal(mu, sig**2)
                kernel += np.reshape(rv.pdf(X),
                                     (int(n / part),
                                     int(n / part),
                                     int(n / part)))

            heights = grsc * np.sin(np.deg2rad(60))
            heights = [heights, -heights]
            ang2 = np.linspace(np.pi / 2, np.pi/2 + 2*np.pi, 3, endpoint=False)
            for h in heights:
                means = zip(
                    grsc * np.cos(np.deg2rad(60)) * np.cos(ang2 * np.sign(h)),
                    grsc * np.cos(np.deg2rad(60)) * np.sin(ang2 * np.sign(h)),
                    np.ones(3) * h
                )
                for mu in means:
                    rv = sc.stats.multivariate_normal(mu, sig**2)
                    kernel += np.reshape(rv.pdf(X),
                                         (int(n / part),
                                         int(n / part),
                                         int(n / part)))

        else:
            raise Exception('What are you doing?')

        res_re = sc.signal.convolve(randfield1, kernel, 'same')
        res_re_2 = sc.signal.convolve(randfield1_2, kernel, 'same')
        res_im = sc.signal.convolve(randfield2, kernel, 'same')
        res_im_2 = sc.signal.convolve(randfield2_2, kernel, 'same')
        ph1 = (np.angle(res_re + 1j*res_im) + np.pi)/2/np.pi
        ph2 = (np.angle(res_re_2 + 1j*res_im_2) + np.pi)/2/np.pi

    return ph1, ph2, x, y, z, kernel


# filtering real and imaginary part of random complex numbers independently, 3D
def create_randomfield(n, sigma):
    # n = 100
    # draw phase of complex number
    rand_ph_help = 2*np.pi*np.random.rand(n, n, n)
    # draw phase of complex number
    rand_ph_help2 = 2*np.pi*np.random.rand(n, n, n)
    randfield1 = np.cos(rand_ph_help)
    randfield1_2 = np.cos(rand_ph_help2)
    randfield2 = np.sin(rand_ph_help)
    randfield2_2 = np.sin(rand_ph_help2)
    x, y, z = np.meshgrid(
        np.linspace(0, 3, n),
        np.linspace(0, 3, n),
        np.linspace(0, 3, n))  # (3mm)**3 voxel

    part = 1
    gridx, gridy, gridz = np.meshgrid(
        np.linspace(-1.5 / part, 1.5 / part, int(n / part)),
        np.linspace(-1.5 / part, 1.5 / part, int(n / part)),
        np.linspace(-1.5 / part, 1.5 / part, int(n / part)))
    X = np.array(
        list(
            zip(
                np.reshape(gridx, (1, -1))[0],
                np.reshape(gridy, (1, -1))[0],
                np.reshape(gridz, (1, -1))[0]
            )
        )
    )
    mu = np.zeros(3)
    sig = np.array([[sigma, 0, 0], [0, sigma, 0], [0, 0, sigma]])**2
    rv = sc.stats.multivariate_normal(mu, sig)
    kernel = np.reshape(rv.pdf(X), (int(n/part), int(n/part), int(n/part)))

    # plt.figure()
    # plt.imshow(kernel[:,:,5])

    res_re = sc.signal.convolve(
        randfield1, kernel, 'same')  # use valid instead
    res_re_2 = sc.signal.convolve(randfield1_2, kernel, 'same')
    res_im = sc.signal.convolve(randfield2, kernel, 'same')
    res_im_2 = sc.signal.convolve(randfield2_2, kernel, 'same')
    ph1 = (np.angle(res_re + 1j*res_im) + np.pi)/2/np.pi
    ph2 = (np.angle(res_re_2 + 1j*res_im_2) + np.pi)/2/np.pi
    return ph1, ph2, x, y, z


def reproduce_gu(ph1, ph2, x, y, z, reps=100000000):
    # pick 10^6 random pairs and measure the pairwise distances
    xf, yf, zf = x.flatten(), y.flatten(), z.flatten()
    phf1, phf2 = ph1.flatten(), ph2.flatten()
    ind1 = np.random.randint(0, len(xf), reps)
    ind2 = np.random.randint(0, len(xf), reps)
    xdist = np.sqrt(
        (xf[ind1] - xf[ind2]) ** 2 + (yf[ind1] - yf[ind2]) ** 2 +
        (zf[ind1] - zf[ind2]) ** 2)
    phfr1, phfr2 = convert_to_rhombus(phf1, phf2)
    phdist2 = circdiff(phfr2[ind1], phfr2[ind2], np.sqrt(3)/2)
    phdist1 = circdiff(phfr1[ind1], phfr1[ind2], 1. + phdist2/np.sqrt(3))
    phdistr1, phdistr2 = convert_to_rhombus(phdist1, phdist2)
    phdist = np.sqrt(phdistr1**2 + phdistr2**2)

    bins = np.linspace(0.01, 0.5, 50, endpoint=True)
    meandiff = np.zeros(len(bins)-1)
    stddiff = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        meandiff[i] = np.mean(phdist[(xdist > bins[i]) * (xdist < bins[i+1])])
        stddiff[i] = np.std(phdist[(xdist > bins[i]) * (xdist < bins[i+1])])

    return meandiff, stddiff, (bins[:-1]+bins[1:])/2


def reproduce_gu2(ph1, ph2, x, y, z, reps=100000000):
    # pick 10^6 random pairs and measure the pairwise distances
    xf, yf, zf = x.flatten(), y.flatten(), z.flatten()
    phf1, phf2 = ph1.flatten(), ph2.flatten()
    ind1 = np.random.randint(0, len(xf), reps)
    ind2 = np.random.randint(0, len(xf), reps)
    xdist = np.sqrt(
        (xf[ind1] - xf[ind2]) ** 2 + (yf[ind1] - yf[ind2]) ** 2 +
        (zf[ind1] - zf[ind2]) ** 2)

    xs = np.linspace(-1, 1, 3, endpoint=True)
    ys = np.linspace(-1, 1, 3, endpoint=True)
    XS, YS = np.meshgrid(xs, ys)
    xsf, ysf = XS.flatten(), YS.flatten()

    phfr1, phfr2 = convert_to_rhombus(phf1, phf2)
    xsfr, ysfr = convert_to_rhombus(xsf, ysf)

    phdist1 = phfr1[ind1].reshape(
        -1,
        1
    ) - (phfr1[ind2].reshape(-1, 1) + xsfr.reshape(1, -1))
    phdist2 = phfr2[ind1].reshape(
        -1,
        1
    ) - (phfr2[ind2].reshape(-1, 1) + ysfr.reshape(1, -1))
    phdist = np.amin(np.sqrt(phdist1**2 + phdist2**2), axis=-1)

    bins = np.linspace(0.01, 0.5, 50, endpoint=True)
    meandiff = np.zeros(len(bins)-1)
    stddiff = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        meandiff[i] = np.mean(phdist[(xdist > bins[i]) * (xdist < bins[i+1])])
        stddiff[i] = np.std(phdist[(xdist > bins[i]) * (xdist < bins[i+1])])

    return meandiff, stddiff, (bins[:-1]+bins[1:])/2


rfield_kappas_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "rfield_kappas.pkl"
)

rep = 300
ngridcells = 200
realncells = 20


def mfunc(i):
    ph1n, ph2n, x, y, z, kerneln = create_randomfield_3D(
        n=ngridcells, mode="none")
    ph1, ph2, x, y, z, kernel = create_randomfield_3D(
        n=ngridcells, mode='grid')
    ph1c, ph2c, xc, yc, zc, kernelc = create_randomfield_3D(
        n=ngridcells, mode='gauss')
    kx_none, ky_none = vonmises.fit(
        2 * np.pi * np.ravel(ph1n),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.ravel(ph2n),
        fscale=1)[0]
    kx_grid, ky_grid = vonmises.fit(
        2 * np.pi * np.ravel(ph1),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.ravel(ph2),
        fscale=1)[0]
    kx_gauss, ky_gauss = vonmises.fit(
        2 * np.pi * np.ravel(ph1c),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.ravel(ph2c),
        fscale=1)[0]

    kx_none, ky_none = vonmises.fit(
        2 * np.pi * np.random.choice(
            np.ravel(ph1n),
            size=int(realncells ** 3),
            replace=False),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.random.choice(
            np.ravel(ph2n),
            size=int(realncells ** 3),
            replace=False),
        fscale=1)[0]
    kx_grid, ky_grid = vonmises.fit(
        2 * np.pi * np.random.choice(
            np.ravel(ph1),
            size=int(realncells ** 3),
            replace=False),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.random.choice(
            np.ravel(ph2),
            size=int(realncells ** 3),
            replace=False),
        fscale=1)[0]
    kx_gauss, ky_gauss = vonmises.fit(2 * np.pi * np.random.choice(
        np.ravel(ph1c),
        size=int(realncells ** 3),
        replace=False),
        fscale=1)[0], vonmises.fit(2 * np.pi *
                                   np.random.choice(
                                       np.ravel(ph2c),
                                       size=int(realncells ** 3),
                                       replace=False),
                                   fscale=1)[0]

    return kx_none, ky_none, kx_gauss, ky_gauss, kx_grid, ky_grid


if not os.path.isfile(rfield_kappas_fname):
    os.makedirs(os.path.dirname(rfield_kappas_fname), exist_ok=True)
    alldata = Parallel(
        n_jobs=-1, verbose=100)(delayed(mfunc)(i) for i in tqdm(range(rep))
                                )
    alldata = np.moveaxis(np.array(alldata), 0, -1)
    with open(rfield_kappas_fname, 'wb') as f:
        pickle.dump(alldata, f)
else:
    with open(rfield_kappas_fname, "rb") as f:
        alldata = pickle.load(f)

kx_none, ky_none, kx_gauss, ky_gauss, kx_grid, ky_grid = alldata

npointscells = 91
ncellslist = np.linspace(10, 100, npointscells, endpoint=True).astype(int)


def mfunc_nkappa(i):
    kx_none = np.zeros(len(ncellslist))
    ky_none = np.zeros(len(ncellslist))
    kx_gauss = np.zeros(len(ncellslist))
    ky_gauss = np.zeros(len(ncellslist))
    kx_grid = np.zeros(len(ncellslist))
    ky_grid = np.zeros(len(ncellslist))
    ph1n, ph2n, x, y, z, kerneln = create_randomfield_3D(
        n=ngridcells, mode="none")
    ph1, ph2, x, y, z, kernel = create_randomfield_3D(
        n=ngridcells, mode='grid')
    ph1c, ph2c, xc, yc, zc, kernelc = create_randomfield_3D(
        n=ngridcells, mode='gauss')
    for ik, ncells in enumerate(ncellslist):
        kx_none[ik], ky_none[ik] = vonmises.fit(
            2 * np.pi * np.random.choice(
                np.ravel(ph1n),
                size=int(ncells ** 3),
                replace=False),
            fscale=1)[0], vonmises.fit(
            2 * np.pi * np.random.choice(
                np.ravel(ph2n),
                size=int(ncells ** 3),
                replace=False),
            fscale=1)[0]
        kx_grid[ik], ky_grid[ik] = vonmises.fit(
            2 * np.pi * np.random.choice(
                np.ravel(ph1),
                size=int(ncells ** 3),
                replace=False),
            fscale=1)[0], vonmises.fit(
            2 * np.pi * np.random.choice(
                np.ravel(ph2),
                size=int(ncells ** 3),
                replace=False),
            fscale=1)[0]
        kx_gauss[ik], ky_gauss[ik] = vonmises.fit(
            2 * np.pi * np.random.choice(
                np.ravel(ph1c),
                size=int(ncells ** 3),
                replace=False),
            fscale=1)[0], vonmises.fit(
            2 * np.pi * np.random.choice(
                np.ravel(ph2c),
                size=int(ncells ** 3),
                replace=False),
            fscale=1)[0]

    return kx_none, ky_none, kx_gauss, ky_gauss, kx_grid, ky_grid


ncells_kappas_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "ncells_kappas.pkl"
)


if not os.path.isfile(ncells_kappas_fname):
    os.makedirs(os.path.dirname(ncells_kappas_fname), exist_ok=True)
    ncellsdata = Parallel(
        n_jobs=-1,
        verbose=100
    )(delayed(mfunc_nkappa)(i) for i in tqdm(range(rep)))
    ncellsdata = np.moveaxis(np.array(ncellsdata), 0, -1)
    with open(ncells_kappas_fname, 'wb') as f:
        pickle.dump(ncellsdata, f)
else:
    with open(ncells_kappas_fname, "rb") as f:
        ncellsdata = pickle.load(f)

(
    kx_none_ncells,
    ky_none_ncells,
    kx_gauss_ncells,
    ky_gauss_ncells,
    kx_grid_ncells,
    ky_grid_ncells
) = ncellsdata

rfield_none_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "rfield_none.pkl"
)
rfield_gauss_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "rfield_gauss.pkl"
)
rfield_grid_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "rfield_grid.pkl"
)


if not os.path.isfile(rfield_none_fname):
    os.makedirs(os.path.dirname(rfield_none_fname), exist_ok=True)
    nonedata = create_randomfield_3D(n=ngridcells, mode='none')
    with open(rfield_none_fname, 'wb') as f:
        pickle.dump(nonedata, f)
else:
    with open(rfield_none_fname, "rb") as f:
        nonedata = pickle.load(f)


ph1n, ph2n, xn, yn, zn, _ = nonedata


if not os.path.isfile(rfield_gauss_fname):
    os.makedirs(os.path.dirname(rfield_gauss_fname), exist_ok=True)
    gaussdata = create_randomfield_3D(n=ngridcells, mode='gauss')
    with open(rfield_gauss_fname, 'wb') as f:
        pickle.dump(gaussdata, f)
else:
    with open(rfield_gauss_fname, "rb") as f:
        gaussdata = pickle.load(f)


ph1c, ph2c, xc, yc, zc, _ = gaussdata


if not os.path.isfile(rfield_grid_fname):
    os.makedirs(os.path.dirname(rfield_grid_fname), exist_ok=True)
    griddata = create_randomfield_3D(n=ngridcells, mode='grid')
    with open(rfield_grid_fname, 'wb') as f:
        pickle.dump(griddata, f)
else:
    with open(rfield_grid_fname, "rb") as f:
        griddata = pickle.load(f)


ph1, ph2, x, y, z, _ = griddata


kernel_gauss_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "kernel_gauss.pkl"
)
kernel_grid_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "kernel_grid.pkl"
)


nkgrid = 200


if not os.path.isfile(kernel_gauss_fname):
    os.makedirs(os.path.dirname(kernel_gauss_fname), exist_ok=True)
    gausskerneldata = create_randomfield_3D(n=nkgrid, mode='gauss')
    with open(kernel_gauss_fname, 'wb') as f:
        pickle.dump(gausskerneldata, f)
else:
    with open(kernel_gauss_fname, "rb") as f:
        gausskerneldata = pickle.load(f)


_, _, _, _, _, kernelc = gausskerneldata


if not os.path.isfile(kernel_grid_fname):
    os.makedirs(os.path.dirname(kernel_grid_fname), exist_ok=True)
    gridkerneldata = create_randomfield_3D(n=nkgrid, mode='grid')
    with open(kernel_grid_fname, 'wb') as f:
        pickle.dump(gridkerneldata, f)
else:
    with open(kernel_grid_fname, "rb") as f:
        gridkerneldata = pickle.load(f)


_, _, _, _, _, kernel = gridkerneldata


###############################################################################
#   Plotting                                                                  #
###############################################################################
fig = plt.figure(figsize=(23, 14))
spec = fig.add_gridspec(
    ncols=7,
    nrows=5,
    width_ratios=[0.9, 0.9, 0.9, 1, 1, 0.6, 0.6],
    height_ratios=[1, 1, 1, 1, 1.5]
)
plt.rcParams.update({'font.size': settings.fs * 0.9})

ax_fieldnone = fig.add_subplot(spec[0, 0])
ax_fieldnone.spines['top'].set_visible(False)
ax_fieldnone.spines['right'].set_visible(False)
pcolor_psearch = plt.imshow(
    ph1n[:, :, 10], cmap='twilight_shifted', origin="lower", vmin=0, vmax=1
)
# plt.yticks(np.linspace(0, ngridcells, 2, endpoint=True), [])
plt.yticks([ngridcells-1], [3.0])
# plt.xticks(np.linspace(0, ngridcells, 2, endpoint=True), [])
plt.xticks(np.linspace(0, ngridcells, 2, endpoint=True),
           np.linspace(0, ngridcells, 2, endpoint=True)/ngridcells*3)
plt.ylabel("y (mm)")
plt.xlabel("x (mm)")

ax_fieldgauss = fig.add_subplot(spec[1, 0])
ax_fieldgauss.spines['top'].set_visible(False)
ax_fieldgauss.spines['right'].set_visible(False)
plt.imshow(ph1c[:, :, 10], cmap='twilight_shifted', origin="lower")
# plt.yticks(np.linspace(0, ngridcells, 2, endpoint=True), [])
plt.yticks([ngridcells-1], [3.0])
# plt.xticks(np.linspace(0, ngridcells, 2, endpoint=True), [])
plt.xticks(np.linspace(0, ngridcells, 2, endpoint=True),
           np.linspace(0, ngridcells, 2, endpoint=True)/ngridcells*3)
plt.ylabel("y (mm)")
plt.xlabel("x (mm)")

ax_fieldgrid = fig.add_subplot(spec[2, 0])
ax_fieldgrid.spines['top'].set_visible(False)
ax_fieldgrid.spines['right'].set_visible(False)
plt.imshow(ph1[:, :, 10], cmap='twilight_shifted', origin="lower")
plt.yticks([ngridcells-1], [3.0])
plt.xticks(np.linspace(0, ngridcells, 2, endpoint=True),
           np.linspace(0, ngridcells, 2, endpoint=True)/ngridcells*3)
plt.ylabel("y (mm)")
plt.xlabel("x (mm)")

field_axs = [ax_fieldnone, ax_fieldgauss, ax_fieldgrid]
for i, field_ax in enumerate(field_axs):
    div_psearch = make_axes_locatable(field_ax)
    cax_psearch = div_psearch.append_axes('right', size='6.5%', pad=0.05)
    cbar_psearch = fig.colorbar(
        pcolor_psearch,
        cax=cax_psearch,
        fraction=0.020,
        pad=0.04,
        ticks=[0., 1.]
    )
    cbar_psearch.ax.set_yticklabels([r"$0$", r"$2\pi$"])
    if i == 0:
        cbar_psearch.set_label("Phase", fontsize=int(settings.fs))


krange = np.array([50, 150]) * nkgrid / 200


ax_kerngaussxy = fig.add_subplot(spec[1, 1])
ax_kerngaussxy.spines['top'].set_visible(False)
ax_kerngaussxy.spines['right'].set_visible(False)
plt.imshow(np.sum(kernelc, axis=-1).T,
           interpolation="gaussian", origin="lower")
plt.xlim(int(krange[0]), int(krange[1]))
plt.ylim(int(krange[0]), int(krange[1]))
# plt.yticks((int(krange[0]), int(krange[1])), [])
# plt.xticks((int(krange[0]), int(krange[1])), [])
plt.yticks([int(krange[1])], [100 / nkgrid * 3.])
plt.xticks([int(krange[0]), int(krange[1])], [0, 100 / nkgrid * 3.])
plt.ylabel("y (mm)")
plt.xlabel("x (mm)")


ax_kerngaussyz = fig.add_subplot(spec[1, 2])
ax_kerngaussyz.spines['top'].set_visible(False)
ax_kerngaussyz.spines['right'].set_visible(False)
plt.imshow(np.sum(kernelc, axis=0).T, interpolation="gaussian", origin="lower")
plt.xlim(int(krange[0]), int(krange[1]))
plt.ylim(int(krange[0]), int(krange[1]))
# plt.yticks((int(krange[0]), int(krange[1])), [])
# plt.xticks((int(krange[0]), int(krange[1])), [])
plt.yticks([int(krange[1])], [100 / nkgrid * 3.])
plt.xticks([int(krange[0]), int(krange[1])], [0, 100 / nkgrid * 3.])
plt.ylabel("z (mm)")
plt.xlabel("y (mm)")


ax_kerngridxy = fig.add_subplot(spec[2, 1])
ax_kerngridxy.spines['top'].set_visible(False)
ax_kerngridxy.spines['right'].set_visible(False)
plt.imshow(np.sum(kernel, axis=-1).T, interpolation="gaussian", origin="lower")
plt.vlines(
    1.15 / 1.5 * (krange[1] - krange[0]) + krange[0],
    0.75 / 1.5 * (krange[1] - krange[0]) + krange[0],
    1.05 / 1.5 * (krange[1] - krange[0]) + krange[0],
    color="red",
    linewidth=4.,
    zorder=100
)
plt.xlim(int(krange[0]), int(krange[1]))
plt.ylim(int(krange[0]), int(krange[1]))
plt.yticks([int(krange[1])], [100 / nkgrid * 3.])
plt.xticks([int(krange[0]), int(krange[1])], [0, 100 / nkgrid * 3.])
plt.ylabel("y (mm)")
plt.xlabel("x (mm)")


ax_kerngridyz = fig.add_subplot(spec[2, 2])
ax_kerngridyz.spines['top'].set_visible(False)
ax_kerngridyz.spines['right'].set_visible(False)
plt.imshow(np.sum(kernel, axis=0).T, interpolation="gaussian", origin="lower")
plt.hlines(
    0.35 / 1.5 * (krange[1] - krange[0]) + krange[0],
    0.75 / 1.5 * (krange[1] - krange[0]) + krange[0],
    1.05 / 1.5 * (krange[1] - krange[0]) + krange[0],
    color="red",
    linewidth=4.,
    zorder=100
)
plt.xlim(int(krange[0]), int(krange[1]))
plt.ylim(int(krange[0]), int(krange[1]))
# plt.yticks((int(krange[0]), int(krange[1])), [])
# plt.xticks((int(krange[0]), int(krange[1])), [])
plt.yticks([int(krange[1])], [100 / nkgrid * 3.])
plt.xticks([int(krange[0]), int(krange[1])], [0, 100 / nkgrid * 3.])
plt.ylabel("z (mm)")
plt.xlabel("y (mm)")


pairw_none_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "pairw_none.pkl"
)
pairw_gauss_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "pairw_gauss.pkl"
)
pairw_grid_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "pairw_grid.pkl"
)

preps = 100000000

# pairwise phase distances
if not os.path.isfile(pairw_none_fname):
    os.makedirs(os.path.dirname(pairw_none_fname), exist_ok=True)
    meandiff, stddiff, bins = reproduce_gu2(ph1n, ph2n, xn, yn, zn, reps=preps)
    with open(pairw_none_fname, 'wb') as f:
        pickle.dump((meandiff, stddiff, bins), f)
else:
    with open(pairw_none_fname, "rb") as f:
        meandiff, stddiff, bins = pickle.load(f)
ax_pairnone = fig.add_subplot(spec[0, 3:5])
ax_pairnone.spines['top'].set_visible(False)
ax_pairnone.spines['right'].set_visible(False)
plt.errorbar(bins * 1000, meandiff, stddiff / np.sqrt(preps), color="black")
# plt.scatter(xdist*1000, phdist)
# plt.xlabel('Pairwise anatomical distance (um)')
plt.ylabel('Pairwise distance\nbetween grid\nphase offsets')
plt.xlabel("Pairwise anatomical distance ($\\mu m$)")
plt.xlim(0, 520)
plt.ylim(0.1, 0.4)
# plt.xticks(np.linspace(0, 500, 6, endpoint=True), [])
plt.xticks(np.linspace(0, 500, 6, endpoint=True),
           np.linspace(0, 500, 6, endpoint=True).astype(int))
# plt.yticks([0.1, 0.4], np.array([0.1, 0.4]) * 30)
plt.yticks([0.1, 0.4], np.array([0.1, 0.4]))


if not os.path.isfile(pairw_gauss_fname):
    os.makedirs(os.path.dirname(pairw_gauss_fname), exist_ok=True)
    meandiff, stddiff, bins = reproduce_gu2(ph1c, ph2c, xc, yc, zc, reps=preps)
    with open(pairw_gauss_fname, 'wb') as f:
        pickle.dump((meandiff, stddiff, bins), f)
else:
    with open(pairw_gauss_fname, "rb") as f:
        meandiff, stddiff, bins = pickle.load(f)
ax_pairgauss = fig.add_subplot(spec[1, 3:5])
ax_pairgauss.spines['top'].set_visible(False)
ax_pairgauss.spines['right'].set_visible(False)
plt.errorbar(bins * 1000, meandiff, stddiff / np.sqrt(preps), color="black")
# plt.scatter(xdist*1000, phdist)
plt.ylabel('Pairwise distance\nbetween grid\nphase offsets')
plt.xlabel("Pairwise anatomical distance ($\\mu m$)")
plt.xlim(0, 520)
plt.ylim(0.1, 0.4)
# plt.xticks(np.linspace(0, 500, 6, endpoint=True), [])
plt.xticks(np.linspace(0, 500, 6, endpoint=True),
           np.linspace(0, 500, 6, endpoint=True).astype(int))
# plt.yticks([0.1, 0.4], np.array([0.1, 0.4]) * 30)
plt.yticks([0.1, 0.4], np.array([0.1, 0.4]))


if not os.path.isfile(pairw_grid_fname):
    os.makedirs(os.path.dirname(pairw_grid_fname), exist_ok=True)
    meandiff, stddiff, bins = reproduce_gu2(ph1, ph2, x, y, z, reps=preps)
    with open(pairw_grid_fname, 'wb') as f:
        pickle.dump((meandiff, stddiff, bins), f)
else:
    with open(pairw_grid_fname, "rb") as f:
        meandiff, stddiff, bins = pickle.load(f)
ax_pairgrid = fig.add_subplot(spec[2, 3:5])
ax_pairgrid.spines['top'].set_visible(False)
ax_pairgrid.spines['right'].set_visible(False)
plt.errorbar(bins * 1000, meandiff, stddiff / np.sqrt(preps), color="black")
# plt.scatter(xdist*1000, phdist)
plt.ylabel('Pairwise distance\nbetween grid\nphase offsets')
plt.xlabel("Pairwise anatomical distance ($\\mu m$)")
plt.xlim(0, 520)
plt.ylim(0.1, 0.4)
plt.xticks(np.linspace(0, 500, 6, endpoint=True),
           np.linspace(0, 500, 6, endpoint=True).astype(int))
# plt.yticks([0.1, 0.4], np.array([0.1, 0.4]) * 30)
plt.yticks([0.1, 0.4], np.array([0.1, 0.4]))
# plt.xlabel("Pairwise anatomical distance ($\mu m$)")


# inset axes....
axins = ax_pairgrid.inset_axes([0.275, 0.1, 0.6, 0.5])
axins.errorbar(bins * 1000, meandiff, stddiff / np.sqrt(preps), color="black")
# sub region of the original image
axins.set_xlim(100, 500)
axins.set_ylim(0.335, 0.360)
axins.set_xticks([])
axins.set_yticks([])
axins.set_xticklabels([])
axins.set_yticklabels([])
ax_pairgrid.indicate_inset_zoom(axins, edgecolor="black")


res = 101
x = np.linspace(0, 1, res, endpoint=True)
y = np.linspace(0, 1, res, endpoint=True)
X, Y = np.meshgrid(x, y)
Xc, Yc = convert_to_rhombus(X, Y)


h_none, _, _ = np.histogram2d(
    ph1n.flatten(),
    ph2n.flatten(),
    bins=(Xc[0, :],
          Yc[:, 0]))
h_gauss, _, _ = np.histogram2d(
    ph1c.flatten(), ph2c.flatten(), bins=(Xc[0, :], Yc[:, 0]))
h_grid, _, _ = np.histogram2d(
    ph1.flatten(), ph2.flatten(), bins=(Xc[0, :], Yc[:, 0]))


cbarmin = np.amin(np.array([h_none, h_gauss, h_grid]))
cbarmax = np.amax(np.array([h_none, h_gauss, h_grid]))


phases_eg_fname = os.path.join(
    settings.loc,
    "clustering",
    "random_fields",
    "phases_eg.pkl"
)


if not os.path.isfile(phases_eg_fname):
    os.makedirs(os.path.dirname(phases_eg_fname), exist_ok=True)
    kx_none_phases, ky_none_phases = vonmises.fit(
        2 * np.pi * np.ravel(ph1n),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.ravel(ph2n),
        fscale=1)[0]
    kx_grid_phases, ky_grid_phases = vonmises.fit(
        2 * np.pi * np.ravel(ph1),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.ravel(ph2),
        fscale=1)[0]
    kx_gauss_phases, ky_gauss_phases = vonmises.fit(
        2 * np.pi * np.ravel(ph1c),
        fscale=1)[0], vonmises.fit(
        2 * np.pi * np.ravel(ph2c),
        fscale=1)[0]
    with open(phases_eg_fname, 'wb') as f:
        pickle.dump([kx_none_phases, ky_none_phases, kx_grid_phases,
                    ky_grid_phases, kx_gauss_phases, ky_gauss_phases], f)
else:
    with open(phases_eg_fname, "rb") as f:
        (
            kx_none_phases,
            ky_none_phases,
            kx_grid_phases,
            ky_grid_phases,
            kx_gauss_phases,
            ky_gauss_phases
        ) = pickle.load(f)


ax_phasesnone = fig.add_subplot(spec[0, 5:])
im = ax_phasesnone.pcolormesh(
    Xc, Yc, h_none, shading='auto', vmin=cbarmin, vmax=cbarmax)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax_phasesnone.set_aspect('equal')
ax_phasesnone.axis('off')
divider = make_axes_locatable(ax_phasesnone)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, ticks=[cbarmin, cbarmax])
cb.ax.set_yticklabels(["min", "max"])
cb.set_label('Density', rotation=90)
plt.text(
    0.85, 0.76,
    f"""
    $\\kappa_s$={
        np.round_(np.mean(np.array([kx_none_phases, ky_none_phases])), 3)
    }
    """,
    fontsize=settings.fs, transform=plt.gcf().transFigure, zorder=100)


ax_phasesgauss = fig.add_subplot(spec[1, 5:])
im = ax_phasesgauss.pcolormesh(
    Xc, Yc, h_gauss, shading='auto', vmin=cbarmin, vmax=cbarmax)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax_phasesgauss.set_aspect('equal')
ax_phasesgauss.axis('off')
divider = make_axes_locatable(ax_phasesgauss)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, ticks=[cbarmin, cbarmax])
cb.ax.set_yticklabels(["min", "max"])
cb.set_label('Density', rotation=90)
plt.text(
    0.85, 0.605,
    f"""
    $\\kappa_s$={
        np.round_(np.mean(np.array([kx_gauss_phases, ky_gauss_phases])), 3)
    }
    """,
    fontsize=settings.fs, transform=plt.gcf().transFigure, zorder=100)


ax_phasesgrid = fig.add_subplot(spec[2, 5:])
im = ax_phasesgrid.pcolormesh(
    Xc, Yc, h_grid, shading='auto', vmin=cbarmin, vmax=cbarmax)
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--')
plt.plot([0.5, 1.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'k--')
plt.plot([1, 1.5], [0, np.sqrt(3)/2], 'k--')
ax_phasesgrid.set_aspect('equal')
ax_phasesgrid.axis('off')
divider = make_axes_locatable(ax_phasesgrid)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(im, cax=cax, ticks=[cbarmin, cbarmax])
cb.ax.set_yticklabels(["min", "max"])
cb.set_label('Density', rotation=90)
plt.text(
    0.85, 0.445,
    f"""
    $\\kappa_s$={
        np.round_(np.mean(np.array([kx_grid_phases, ky_grid_phases])), 3)
    }
    """,
    fontsize=settings.fs, transform=plt.gcf().transFigure, zorder=100)


(
    kx_none_ncells,
    ky_none_ncells,
    kx_gauss_ncells,
    ky_gauss_ncells,
    kx_grid_ncells,
    ky_grid_ncells
) = ncellsdata


# ncells kappas
ncellslist = np.linspace(10, 100, npointscells, endpoint=True).astype(int)
ax_ncells = fig.add_subplot(spec[3:5, 0:3])
ax_ncells.spines['top'].set_visible(False)
ax_ncells.spines['right'].set_visible(False)
k_none_ncells = np.hstack((kx_none_ncells, ky_none_ncells))
k_gauss_ncells = np.hstack((kx_gauss_ncells, ky_gauss_ncells))
k_grid_ncells = np.hstack((kx_grid_ncells, ky_grid_ncells))
plt.plot(ncellslist**3, np.mean(k_none_ncells, axis=-1),
         color="black", label="no kernel")
plt.plot(ncellslist ** 3, np.mean(k_gauss_ncells, axis=-1),
         color="black", linestyle="--", label="gaussian kernel")
plt.plot(ncellslist ** 3, np.mean(k_grid_ncells, axis=-1),
         color="black", linestyle=":", linewidth=3, label="grid kernel")
plt.plot(ncellslist ** 3, 1 / np.sqrt(ncellslist ** 3),
         color="black", linestyle="dashdot", label="$1 / \\sqrt{N}$")
plt.legend(prop={'size': 0.65 * settings.fs}, loc="lower left")
plt.xlim(10**3, 100**3)
plt.vlines(20**3, 1e-3, 1e-1, linestyle=":", color="black")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of cells, N", labelpad=3)
plt.ylabel(r"Concentration parameter $\kappa_s$")
ax_ncells.set_aspect('equal', adjustable=None)


def tick_function(X):
    V = (3/(X**(1/3))) * 1000
    return ["%.0f" % z for z in V]


ax_ncells2 = fig.add_axes(ax_ncells.get_position())
plt.plot(ncellslist**3, np.mean(k_none_ncells, axis=-1),
         color="black", label="no kernel", alpha=0)
plt.plot(ncellslist ** 3, np.mean(k_gauss_ncells, axis=-1),
         color="black", linestyle="--", label="gaussian kernel", alpha=0)
plt.plot(ncellslist**3, np.mean(k_grid_ncells, axis=-1), color="black",
         linestyle=":", linewidth=3, label="grid kernel", alpha=0)
plt.plot(ncellslist**3, 1 / np.sqrt(ncellslist**3), color="black",
         linestyle="dashdot", label="$1 / \\sqrt{N}$", alpha=0)
ax_ncells2.set_facecolor("None")
ax_ncells2.set_aspect('equal')
ax_ncells2.xaxis.set_ticks_position("bottom")
ax_ncells2.xaxis.set_label_position("bottom")
ax_ncells2.spines["bottom"].set_position(("axes", -0))
ax_ncells2.set_frame_on(True)
ax_ncells2.patch.set_visible(False)
ncellslist2 = np.array([10**3, 20**3, 50**3, 100**3])
ax_ncells2.set_xscale("log")
ax_ncells2.set_xlim(10**3, 100**3)
ax_ncells2.set_xticks(ncellslist2)
ax_ncells2.set_xticklabels(tick_function(ncellslist2))
ax_ncells2.set_xlabel(r"Average distance to nearest neighbour ($\mu m$)")
plt.minorticks_off()
plt.yticks([])

for sp in ax_ncells2.spines.values():
    sp.set_visible(False)
ax_ncells2.spines["bottom"].set_visible(True)

# kappas
kpos = [1, 1.5, 2]
ckpos = [np.mean(kpos[:2]), np.mean(kpos), np.mean(kpos[1:])]
k_none = np.hstack((kx_none, ky_none))
k_gauss = np.hstack((kx_gauss, ky_gauss))
k_grid = np.hstack((kx_grid, ky_grid))
ax_kappas = fig.add_subplot(spec[3:5, 3:5])
ax_kappas.spines['top'].set_visible(False)
ax_kappas.spines['right'].set_visible(False)
plt.violinplot(
    [
        k_none,
        k_gauss,
        k_grid
    ],
    kpos,
    showmedians=True,
    widths=0.4
)
plt.xticks(
    kpos,
    ['No\nkernel', 'Gaussian\nkernel', 'Grid\nkernel']
)
plt.ylim(0, 0.16)
plt.yticks([0., 0.08, 0.16])
plt.ylabel(r"Concentration parameter $\kappa_s$")


pvals = [
    mannwhitneyu(k_none, k_gauss).pvalue,
    mannwhitneyu(k_none, k_grid).pvalue,
    mannwhitneyu(k_gauss, k_grid).pvalue
]


def mannwhitneysig(pval):
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"


plt.text(ckpos[0], 0.090, f"{mannwhitneysig(pvals[0])}",
         fontsize=settings.fs, zorder=100, ha="center")
plt.plot(np.array([kpos[0], kpos[1]]), np.array(
    [0.085, 0.085]), color="black", linewidth=1.5)
plt.plot(np.array([kpos[0], kpos[0] + 1e-6]),
         np.array([0.075, 0.085]), color="black", linewidth=1.5)
plt.plot(np.array([kpos[1], kpos[1] + 1e-6]),
         np.array([0.075, 0.085]), color="black", linewidth=1.5)
plt.plot(np.array([ckpos[0], ckpos[0] + 1e-6]),
         np.array([0.085, 0.090]), color="black", linewidth=1.5)

plt.text(ckpos[1], 0.16, f"{mannwhitneysig(pvals[1])}",
         fontsize=settings.fs, zorder=100, ha="center")
plt.plot(np.array([kpos[0], kpos[2]]), np.array(
    [0.155, 0.155]), color="black", linewidth=1.5)
plt.plot(np.array([kpos[0], kpos[0] + 1e-6]),
         np.array([0.145, 0.155]), color="black", linewidth=1.5)
plt.plot(np.array([kpos[2], kpos[2] + 1e-6]),
         np.array([0.145, 0.155]), color="black", linewidth=1.5)
plt.plot(np.array([ckpos[1], ckpos[1] + 1e-6]),
         np.array([0.155, 0.16]), color="black", linewidth=1.5)

plt.text(ckpos[2], 0.135, f"{mannwhitneysig(pvals[2])}",
         fontsize=settings.fs, zorder=100, ha="center")
plt.plot(np.array([kpos[1], kpos[2]]), np.array(
    [0.13, 0.13]), color="black", linewidth=1.5)
plt.plot(np.array([kpos[1], kpos[1] + 1e-6]),
         np.array([0.12, 0.13]), color="black", linewidth=1.5)
plt.plot(np.array([kpos[2], kpos[2] + 1e-6]),
         np.array([0.12, 0.13]), color="black", linewidth=1.5)
plt.plot(np.array([ckpos[2], ckpos[2] + 1e-6]),
         np.array([0.13, 0.135]), color="black", linewidth=1.5)

###############################################################################
# Plotting tweaks                                                             #
###############################################################################
plt.subplots_adjust(
    wspace=1.25,
    hspace=1.25
)
ax_pos(ax_fieldnone, 0, 0, 1.4, 1.4)
ax_pos(ax_fieldgauss, 0, 0.01, 1.4, 1.4)
ax_pos(ax_fieldgrid, 0, 0.02, 1.4, 1.4)

ax_pos(ax_kerngaussxy, 0.03, 0.01, 1.4, 1.4)
ax_pos(ax_kerngaussyz, 0.01, 0.01, 1.4, 1.4)
ax_pos(ax_kerngridxy, 0.03, 0.02, 1.4, 1.4)
ax_pos(ax_kerngridyz, 0.01, 0.02, 1.4, 1.4)

ax_pos(ax_pairnone, 0.03, 0, 0.9, 1.4)
ax_pos(ax_pairgauss, 0.03, 0.01, 0.9, 1.4)
ax_pos(ax_pairgrid, 0.03, 0.02, 0.9, 1.4)

ax_pos(ax_phasesnone, -0.02, 0, 1.3, 1.3)
ax_pos(ax_phasesgauss, -0.02, 0.01, 1.3, 1.3)
ax_pos(ax_phasesgrid, -0.02, 0.02, 1.3, 1.3)

ax_pos(ax_ncells, -0.04, 0, 1, 0.85)
ax_pos(ax_ncells2, -0.0575, -0.2, 0.703, 1)
ax_pos(ax_kappas, -0.07, -0.04, 1, 1)
plt.savefig(saveloc, dpi=300)
